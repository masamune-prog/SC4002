"""Mixed-architecture joint ensemble with training-lexicon memories for MASSIVE en-US.

Built on DanFeng's Stage-2 recipe (branches DanFeng-both / DanFeng-span, tools/core.py): clitic units, a shared
frozen GloVe + fastText table, trainable unit embeddings, char-CNN, hashed char n-grams, hierarchical scenario head,
intent-aware slots, BIO-constrained CRF, span classifier, SWA, label smoothing. Data parsing, scoring, CRF and the
span decoder are copied from there unchanged so every number is scored with the same rule.

New in this file:
  * encoders: GLU dilated CNN (DanFeng), 2-layer BiLSTM (Max-Stage1), pre-LN Transformer with T5 relative position
    buckets (WL_branch). All share one input layer and one frozen word-vector table.
  * lexicon memories (Engram-style lookups whose rows are counts fitted on the train split, not learned weights):
      - slot lexicon: word n-gram -> how often it was each slot type / how often it occurred at all in train;
        read per token (B/I roles) at the input and again before the slot head, and per span by the span head.
      - intent lexicon: word 1-2-gram -> intent counts; turned into a naive-Bayes style log-distribution that the
        intent head reads.
    For train utterances the utterance's own contribution is subtracted (leave-one-out), so the lookup never
    contains the answer, and the whole memory is dropped for a fraction of training utterances.
  * MixedEnsemble: averages intent probabilities and combines slots by a weighted span-level vote, so CRF and
    span members can be mixed.
Only `utt` is ever used as model input; labels of the train split fit parameters and lexicon tables only.
"""
import os, re, json, time, math, random, copy, zlib, pickle
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASSIVE_PATH = os.path.join(ROOT, "1.0", "data", "en-US.jsonl")
EMB_DIR = os.path.join(ROOT, "embeddings")
GLOVE_PATH = os.path.join(EMB_DIR, "glove.6B.300d.txt")
FASTTEXT_PATH = os.path.join(EMB_DIR, "crawl-300d-2M.vec")
CACHE_DIR = os.path.join(ROOT, "ens", "cache")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------- data + scoring
SLOT_RE = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]*?)\s*\]")


def parse_annot(annot_utt):
    """'wake me up at [time : five am]' -> tokens, BIO tags, gold set of (slot_type, span_text)."""
    tokens, tags, spans, pos = [], [], set(), 0
    for m in SLOT_RE.finditer(annot_utt):
        outside = annot_utt[pos:m.start()].split()
        tokens += outside
        tags += ["O"] * len(outside)
        slot_type, words = m.group(1).strip(), m.group(2).split()
        tokens += words
        tags += [f"B-{slot_type}"] + [f"I-{slot_type}"] * (len(words) - 1)
        spans.add((slot_type, " ".join(words)))
        pos = m.end()
    outside = annot_utt[pos:].split()
    tokens += outside
    tags += ["O"] * len(outside)
    return tokens, tags, spans


def to_example(r):
    tokens, tags, spans = parse_annot(r["annot_utt"])
    assert tokens == r["utt"].split(), f"annot_utt and utt disagree for id {r['id']}"
    return {"id": r["id"], "utt": r["utt"], "tokens": tokens, "tags": tags, "gold_spans": spans,
            "intent": r["intent"], "scenario": r["intent"].split("_", 1)[0]}


def load_data():
    with open(MASSIVE_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    raw = {n: [r for r in records if r["partition"] == n] for n in ("train", "dev", "test")}
    assert len(records) == 16_521
    assert (len(raw["train"]), len(raw["dev"]), len(raw["test"])) == (11_514, 2_033, 2_974)
    return {n: [to_example(r) for r in raw[n]] for n in raw}


def tags_to_spans(tokens, tags):
    """BIO tags -> set of (slot_type, span_text). An I-x that does not continue an x span starts a new span."""
    spans, cur_type, cur_words = set(), None, []
    for tok, tag in zip(tokens, tags):
        if tag == "O":
            if cur_type:
                spans.add((cur_type, " ".join(cur_words)))
            cur_type, cur_words = None, []
        elif tag.startswith("B-") or tag[2:] != cur_type:
            if cur_type:
                spans.add((cur_type, " ".join(cur_words)))
            cur_type, cur_words = tag[2:], [tok]
        else:
            cur_words.append(tok)
    if cur_type:
        spans.add((cur_type, " ".join(cur_words)))
    return spans


def score(pred_intents, pred_spans, examples):
    """The one scoring rule used for every experiment (identical to DanFeng's)."""
    tp = fp = fn = 0
    for pred, e in zip(pred_spans, examples):
        gold = e["gold_spans"]
        tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)
    correct = sum(p == e["intent"] for p, e in zip(pred_intents, examples))
    acc = 100.0 * correct / len(examples)
    f1 = 100.0 * 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 100.0
    return {"intent_acc": acc, "slot_f1": f1, "combined": (acc + f1) / 2, "intent_correct": correct,
            "tp": tp, "fp": fp, "fn": fn}


# ----------------------------------------------------------------------------------------------- vocabularies
PAD, UNK = "<pad>", "<unk>"


class Vocab:
    def __init__(self, items, specials=(PAD, UNK)):
        self.itos = list(specials) + [x for x in items if x not in specials]
        self.stoi = {x: i for i, x in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def get(self, x):
        return self.stoi.get(x, self.stoi.get(UNK, 0))


def by_frequency(counter):
    return [w for w, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]


class Labels:
    """Intent / scenario / BIO tag inventories, fitted on train only."""

    def __init__(self, train):
        self.intents = Vocab(sorted({e["intent"] for e in train}), specials=())
        self.scenarios = Vocab(sorted({e["scenario"] for e in train}), specials=())
        self.slot_types = sorted({t[2:] for e in train for t in e["tags"] if t != "O"})
        self.tags = Vocab(["O"] + [f"{p}-{s}" for s in self.slot_types for p in ("B", "I")], specials=())
        self.i2s = torch.tensor([self.scenarios.stoi[i.split("_", 1)[0]] for i in self.intents.itos])
        self.tag_type = torch.tensor([0] + [self.slot_types.index(t[2:]) + 1 for t in self.tags.itos[1:]])
        self.tag_is_b = torch.tensor([t.startswith("B-") for t in self.tags.itos])


# ----------------------------------------------------------------------------------------------- word vectors
def vector_keys(w):
    s = w.strip(".,!?;:\"()")
    keys = [w, s, s.replace(".", "")]
    m = re.fullmatch(r"(.+?)(n't|'s|'re|'ve|'ll|'d|'m|')", s)
    if m:
        keys.append(tuple(p for p in m.groups() if p))
    parts = [p for p in re.split(r"[-./&]", s) if p]
    if len(parts) > 1:
        keys.append(tuple(parts))
    return keys


def read_vectors(path, wanted, top_k=0):
    vecs, top = {}, []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for rank, line in enumerate(f):
            word, _, rest = line.rstrip("\n").partition(" ")
            if rank == 0 and rest.count(" ") < 5:
                continue
            if word in wanted or len(top) < top_k:
                v = np.asarray(rest.split(" "), dtype=np.float32)
                if v.shape[0] != 300:
                    continue
                vecs[word] = v
                if len(top) < top_k:
                    top.append(word)
    return vecs, top


def lookup_vector(w, vecs):
    for k in vector_keys(w):
        if isinstance(k, tuple):
            if all(p in vecs for p in k):
                return np.mean([vecs[p] for p in k], axis=0)
        elif k in vecs:
            return vecs[k]
    return None


def needed_keys(words):
    return {p for w in words for k in vector_keys(w) for p in (k if isinstance(k, tuple) else (k,))}


# ----------------------------------------------------------------------------------------------- tokenisation
CLITIC_RE = re.compile(r"^(.+?)(n't|'s|'re|'ve|'ll|'d|'m)$")


def clitic_units(token):
    if re.fullmatch(r"[a-z]\.", token):
        return [token[0]]
    m = CLITIC_RE.match(token)
    if m:
        return [m.group(1), m.group(2)]
    return [token]


def split_units(tokens):
    """Units for the encoder, index of each token's first unit, and the token index of every unit."""
    units, first, owner = [], [], []
    for ti, t in enumerate(tokens):
        first.append(len(units))
        us = clitic_units(t)
        units.extend(us)
        owner.extend([ti] * len(us))
    return units, first, owner


def char_ngrams(unit, buckets, n_min=3, n_max=5, max_grams=40):
    w = f"<{unit}>"
    grams = [w] + [w[i:i + n] for n in range(n_min, n_max + 1) for i in range(len(w) - n + 1)]
    return [1 + zlib.crc32(g.encode("utf-8")) % buckets for g in grams[:max_grams]]


# ----------------------------------------------------------------------------------------------- lexicon memories
MAX_SPAN = 10


class SlotLexicon:
    """Word n-gram -> Counter(slot type -> times it was annotated as that slot in train) and
    occ[ngram] = times the n-gram occurs anywhere in a train utterance. Only n-grams that were a slot at least once
    are stored. The stored entries are counted as (frozen) parameters."""

    def __init__(self, train, labels, max_len=MAX_SPAN):
        self.max_len, self.labels = max_len, labels
        self.slot = defaultdict(Counter)
        for e in train:
            for typ, text in e["gold_spans"]:
                self.slot[text][typ] += 1
        self.occ = Counter()
        for e in train:
            for key in self._ngrams(e["tokens"]):
                if key in self.slot:
                    self.occ[key] += 1

    def _ngrams(self, tokens):
        T = len(tokens)
        for i in range(T):
            for w in range(min(self.max_len, T - i)):
                yield " ".join(tokens[i:i + w + 1])

    def n_entries(self):
        return sum(len(c) for c in self.slot.values()) + len(self.occ)

    def span_probs(self, example, leave_out):
        """{(start, width): {type_id: p}} for every span of the utterance found in the lexicon.
        p = count_as_type / (occurrences + 1); with leave_out the utterance's own counts are removed first."""
        tokens = example["tokens"]
        own_slot = Counter((text, typ) for typ, text in example["gold_spans"]) if leave_out else Counter()
        own_occ = Counter(self._ngrams(tokens)) if leave_out else Counter()
        out = {}
        T = len(tokens)
        for i in range(T):
            for w in range(min(self.max_len, T - i)):
                key = " ".join(tokens[i:i + w + 1])
                c = self.slot.get(key)
                if not c:
                    continue
                n = self.occ[key] - own_occ[key]
                if n <= 0:
                    continue
                probs = {}
                for typ, k in c.items():
                    k -= own_slot[(key, typ)]
                    if k > 0:
                        probs[self.labels.slot_types.index(typ)] = k / (n + 1)
                if probs:
                    out[(i, w)] = probs
        return out


def token_lex_features(span_probs, T, n_types):
    """(T, 2*n_types): max probability that the token begins / is inside a lexicon span of each type."""
    f = np.zeros((T, 2 * n_types), dtype=np.float32)
    for (i, w), probs in span_probs.items():
        for k, p in probs.items():
            f[i, k] = max(f[i, k], p)
            for t in range(i + 1, i + w + 1):
                f[t, n_types + k] = max(f[t, n_types + k], p)
    return f


class IntentLexicon:
    """Word 1-2-gram -> intent counts on train (only n-grams seen at least twice are stored)."""

    def __init__(self, train, labels, min_count=2, alpha=0.1):
        self.labels, self.alpha = labels, alpha
        counts = defaultdict(Counter)
        for e in train:
            for g in set(self._grams(e["tokens"])):
                counts[g][labels.intents.stoi[e["intent"]]] += 1
        self.counts = {g: c for g, c in counts.items() if sum(c.values()) >= min_count}

    @staticmethod
    def _grams(tokens):
        return tokens + [a + " " + b for a, b in zip(tokens, tokens[1:])]

    def n_entries(self):
        return sum(len(c) for c in self.counts.values())

    def features(self, example, leave_out):
        """Naive-Bayes style log-distribution over intents (60,), plus the number of n-grams that hit the table."""
        K = len(self.labels.intents)
        own = self.labels.intents.stoi[example["intent"]] if leave_out else None
        score, hits = np.zeros(K, dtype=np.float64), 0
        for g in set(self._grams(example["tokens"])):
            c = self.counts.get(g)
            if not c:
                continue
            vec = np.zeros(K)
            for k, v in c.items():
                vec[k] = v
            if own is not None:
                vec[own] -= 1
            n = vec.sum()
            if n <= 0:
                continue
            score += np.log((vec + self.alpha) / (n + K * self.alpha)) + np.log(K)
            hits += 1
        if hits:
            score = score - score.max()
            score = score - np.log(np.exp(score).sum())
        else:
            score[:] = -np.log(K)
        return score.astype(np.float32), hits


# ----------------------------------------------------------------------------------------------- features
def build_features(data, labels, top_k=2000, ngram_buckets=5000, word_drop_alpha=0.25, use_cache=True):
    """Vocabularies, frozen word-vector table and lexicons, fitted on the train split only (cached on disk)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"feats_top{top_k}_ng{ngram_buckets}.pkl")
    if use_cache and os.path.exists(path):
        with open(path, "rb") as f:
            feats = pickle.load(f)
    else:
        train = data["train"]
        counts = Counter(u for e in train for t in e["tokens"] for u in clitic_units(t))
        units = Vocab(by_frequency(counts))
        chars = Vocab(sorted({c for u in counts for c in u}))
        print("reading GloVe ...", flush=True)
        glove, glove_top = read_vectors(GLOVE_PATH, needed_keys(set(counts)), top_k=top_k)
        print("reading fastText ...", flush=True)
        fasttext, _ = read_vectors(FASTTEXT_PATH, needed_keys(set(counts)) | set(glove_top))
        frozen_words = Vocab(units.itos[2:] + [w for w in glove_top if w not in units.stoi])
        table = np.zeros((len(frozen_words), 600), dtype=np.float32)
        for j, vecs in enumerate((glove, fasttext)):
            for i, w in enumerate(frozen_words.itos[2:], start=2):
                v = lookup_vector(w, vecs)
                if v is not None:
                    table[i, 300 * j:300 * (j + 1)] = v
        drop_p = torch.zeros(len(units))
        for i, w in enumerate(units.itos[2:], start=2):
            drop_p[i] = word_drop_alpha / (word_drop_alpha + counts[w])
        feats = {"units": units, "chars": chars, "frozen_words": frozen_words, "table": torch.tensor(table),
                 "drop_p": drop_p, "ngram_buckets": ngram_buckets}
        with open(path, "wb") as f:
            pickle.dump(feats, f)
    feats["slot_lex"] = SlotLexicon(data["train"], labels)
    feats["intent_lex"] = IntentLexicon(data["train"], labels)
    allowed = torch.zeros(len(labels.intents), len(labels.tags), dtype=torch.bool)
    allowed[:, 0] = True
    for e in data["train"]:
        i = labels.intents.stoi[e["intent"]]
        for t in e["tags"]:
            if t != "O":
                allowed[i, labels.tags.stoi["B-" + t[2:]]] = allowed[i, labels.tags.stoi["I-" + t[2:]]] = True
    feats["allowed_tags"] = allowed
    return feats


def encode(examples, feats, labels, is_train, max_chars=20):
    """is_train switches the lexicon lookups to leave-one-out."""
    out = []
    n_types = len(labels.slot_types)
    for e in examples:
        units, first, owner = split_units(e["tokens"])
        sp = feats["slot_lex"].span_probs(e, leave_out=is_train)
        ilex, ihits = feats["intent_lex"].features(e, leave_out=is_train)
        out.append({"units": [feats["units"].get(u) for u in units],
                    "frozen": [feats["frozen_words"].get(u) for u in units],
                    "chars": [[feats["chars"].get(c) for c in u[:max_chars]] for u in units],
                    "grams": [char_ngrams(u, feats["ngram_buckets"]) for u in units],
                    "first": first, "owner": owner,
                    "tags": [labels.tags.stoi[t] for t in e["tags"]],
                    "intent": labels.intents.stoi[e["intent"]], "scenario": labels.scenarios.stoi[e["scenario"]],
                    "tok_lex": token_lex_features(sp, len(e["tokens"]), n_types),
                    "span_lex": [(i, w, k, p) for (i, w), probs in sp.items() for k, p in probs.items()],
                    "ilex": np.append(ilex, np.log1p(ihits)).astype(np.float32)})
    return out


def pad2d(rows, pad=0):
    width = max(1, max(len(r) for r in rows))
    out = torch.full((len(rows), width), pad, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r, dtype=torch.long)
    return out


def pad3d(rows, pad=0):
    d1 = max(1, max(len(r) for r in rows))
    d2 = max(1, max((len(c) for r in rows for c in r), default=1))
    out = torch.full((len(rows), d1, d2), pad, dtype=torch.long)
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            out[i, j, :len(c)] = torch.tensor(c, dtype=torch.long)
    return out


def collate(items):
    B = len(items)
    T = max(len(it["tags"]) for it in items)
    n_feat = items[0]["tok_lex"].shape[1]
    n_types = n_feat // 2
    tok_lex = torch.zeros(B, T, n_feat)
    span_lex = torch.zeros(B, T * MAX_SPAN, n_types)
    for b, it in enumerate(items):
        tok_lex[b, :len(it["tags"])] = torch.from_numpy(it["tok_lex"])
        for i, w, k, p in it["span_lex"]:
            span_lex[b, i * MAX_SPAN + w, k] = p
    return {"lengths": torch.tensor([len(it["tags"]) for it in items]),
            "tags": pad2d([it["tags"] for it in items]),
            "first": pad2d([it["first"] for it in items]),
            "owner": pad2d([it["owner"] for it in items]),
            "intent": torch.tensor([it["intent"] for it in items]),
            "scenario": torch.tensor([it["scenario"] for it in items]),
            "units": pad2d([it["units"] for it in items]),
            "frozen": pad2d([it["frozen"] for it in items]),
            "chars": pad3d([it["chars"] for it in items]),
            "grams": pad3d([it["grams"] for it in items]),
            "tok_lex": tok_lex, "span_lex": span_lex,
            "ilex": torch.from_numpy(np.stack([it["ilex"] for it in items]))}


def make_batches(encoded, batch_size, shuffle, seed=0):
    order = list(range(len(encoded)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
        pool = batch_size * 50
        order = [i for s in range(0, len(order), pool)
                 for i in sorted(order[s:s + pool], key=lambda i: len(encoded[i]["units"]))]
    batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return [collate([encoded[i] for i in b]) for b in batches]


def to_device(batch):
    return {k: (v if k == "lengths" else v.to(DEVICE, non_blocking=True)) for k, v in batch.items()}


# ----------------------------------------------------------------------------------------------- CRF (DanFeng)
def crf_log_partition(em, mask, trans, start, end):
    alpha = start + em[:, 0]
    for t in range(1, em.size(1)):
        nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + em[:, t]
        alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
    return torch.logsumexp(alpha + end, dim=1)


def crf_path_score(em, tags, mask, trans, start, end):
    maskf = mask.float()
    s = start[tags[:, 0]] + (em.gather(2, tags.unsqueeze(2)).squeeze(2) * maskf).sum(1)
    s = s + (trans[tags[:, :-1], tags[:, 1:]] * maskf[:, 1:]).sum(1)
    last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
    return s + end[last]


def crf_viterbi(em, mask, trans, start, end):
    score, history = start + em[:, 0], []
    for t in range(1, em.size(1)):
        best, idx = (score.unsqueeze(2) + trans.unsqueeze(0)).max(dim=1)
        score = torch.where(mask[:, t].unsqueeze(1), best + em[:, t], score)
        history.append(idx)
    score = score + end
    best_score, last = score.max(dim=1)
    hist = torch.stack(history, 1).cpu() if history else None
    last, lengths = last.cpu().tolist(), mask.sum(1).cpu().tolist()
    paths = []
    for b, L in enumerate(lengths):
        path = [last[b]]
        for t in range(L - 1, 0, -1):
            path.append(int(hist[b, t - 1, path[-1]]))
        paths.append(path[::-1])
    return paths, best_score


def bio_constraints(tag_itos):
    n = len(tag_itos)
    allowed, start_ok = torch.ones(n, n, dtype=torch.bool), torch.ones(n, dtype=torch.bool)
    for j, tj in enumerate(tag_itos):
        if tj.startswith("I-"):
            start_ok[j] = False
            for i, ti in enumerate(tag_itos):
                allowed[i, j] = ti[2:] == tj[2:] and ti != "O"
    return allowed, start_ok


class CRF(nn.Module):
    def __init__(self, n_tags, constraints=None):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))
        pen_t, pen_s = torch.zeros(n_tags, n_tags), torch.zeros(n_tags)
        if constraints is not None:
            allowed, start_ok = constraints
            pen_t, pen_s = (~allowed).float() * -1e4, (~start_ok).float() * -1e4
        self.register_buffer("pen_trans", pen_t)
        self.register_buffer("pen_start", pen_s)

    def potentials(self):
        return self.trans + self.pen_trans, self.start + self.pen_start, self.end

    def nll(self, em, tags, mask):
        tr, st, en = self.potentials()
        return crf_log_partition(em, mask, tr, st, en) - crf_path_score(em, tags, mask, tr, st, en)


# ----------------------------------------------------------------------------------------------- span helpers
def gold_span_grid(tags, lengths, labels, max_span=MAX_SPAN):
    B, T = tags.shape
    grid = torch.zeros(B, T * max_span, dtype=torch.long)
    tags = tags.cpu()
    for b, L in enumerate(lengths.tolist()):
        start, cur = None, 0
        for t in range(L + 1):
            tid = int(tags[b, t]) if t < L else 0
            typ = int(labels.tag_type[tid])
            if cur and (typ != cur or labels.tag_is_b[tid]):
                if t - start <= max_span:
                    grid[b, start * max_span + (t - 1 - start)] = cur
                cur = 0
            if typ and not cur:
                start, cur = t, typ
    return grid


def best_spans(gains, max_span=MAX_SPAN):
    """Exact choice of non-overlapping spans with the largest total positive gain (weighted interval scheduling)."""
    T = len(gains)
    best, choice = [0.0] * (T + 1), [None] * (T + 1)
    for end in range(1, T + 1):
        best[end], choice[end] = best[end - 1], None
        for w in range(min(max_span, end)):
            i = end - 1 - w
            g = gains[i][w]
            if g > 0 and best[i] + g > best[end]:
                best[end], choice[end] = best[i] + g, (i, w)
    spans, end = [], T
    while end > 0:
        if choice[end] is None:
            end -= 1
        else:
            i, w = choice[end]
            spans.append((i, w))
            end = i
    return spans[::-1], best[T]


# ----------------------------------------------------------------------------------------------- model
class SeqConv(nn.Module):
    def __init__(self, c_in, c_out, width=3, dilation=1):
        super().__init__()
        self.width, self.dilation = width, dilation
        self.lin = nn.Linear(width * c_in, c_out)

    def forward(self, x):
        T, pad = x.size(1), (self.width // 2) * self.dilation
        xp = F.pad(x, (0, 0, pad, pad))
        return self.lin(torch.cat([xp[:, i * self.dilation:i * self.dilation + T] for i in range(self.width)], dim=2))


class CharCNN(nn.Module):
    def __init__(self, n_chars, char_dim=50, filters=100, width=3):
        super().__init__()
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.conv = SeqConv(char_dim, filters, width)
        self.out_dim = filters

    def forward(self, chars):
        B, T, C = chars.shape
        flat = chars.view(B * T, C)
        h = self.conv(self.emb(flat)).masked_fill((flat == 0).unsqueeze(2), -1e4).max(dim=1).values
        return torch.tanh(h).view(B, T, -1)


class GLUBlock(nn.Module):
    def __init__(self, channels, dilation, width=3, dropout=0.3):
        super().__init__()
        self.conv = SeqConv(channels, 2 * channels, width, dilation)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, m):
        return self.norm(x + self.drop(F.glu(self.conv(x), dim=2))) * m


class GLUEncoder(nn.Module):
    """DanFeng: residual gated dilated convolutions."""

    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([GLUBlock(cfg["channels"], d, 3, cfg["dropout"]) for d in cfg["dilations"]])

    def forward(self, x, umask):
        m = umask.unsqueeze(2).float()
        for blk in self.blocks:
            x = blk(x, m)
        return x


class LSTMEncoder(nn.Module):
    """Max-Stage1: 2-layer BiLSTM (here over DanFeng's richer inputs)."""

    def __init__(self, cfg):
        super().__init__()
        C = cfg["channels"]
        self.rnn = nn.LSTM(C, C // 2, num_layers=cfg.get("rnn_layers", 2), batch_first=True, bidirectional=True,
                           dropout=cfg["dropout"])
        self.norm = nn.LayerNorm(C)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x, umask):
        lengths = umask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        return self.norm(x + self.drop(h)) * umask.unsqueeze(2).float()


def rel_bucket(rel, num_buckets=32, max_distance=64):
    """T5-style bidirectional relative-position buckets (WL_branch)."""
    nb = num_buckets // 2
    ret = (rel > 0).long() * nb
    n = rel.abs()
    max_exact = nb // 2
    is_small = n < max_exact
    large = max_exact + (torch.log(n.float().clamp(min=1) / max_exact) / math.log(max_distance / max_exact)
                         * (nb - max_exact)).long()
    return ret + torch.where(is_small, n, large.clamp(max=nb - 1))


class TBlock(nn.Module):
    def __init__(self, d, heads, ffn, dropout, attn_dropout):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = attn_dropout

    def forward(self, x, bias):
        B, U, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, U, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, U, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class TransformerEncoder(nn.Module):
    """WL_branch: pre-LN Transformer with learned T5 relative-position biases."""

    def __init__(self, cfg):
        super().__init__()
        C, H = cfg["channels"], cfg.get("heads", 4)
        self.heads = H
        self.rel = nn.Embedding(32, H)
        self.layers = nn.ModuleList([TBlock(C, H, cfg.get("ffn", 4 * C), cfg["dropout"], 0.1)
                                     for _ in range(cfg.get("layers", 4))])
        self.norm = nn.LayerNorm(C)

    def forward(self, x, umask):
        U = x.size(1)
        pos = torch.arange(U, device=x.device)
        bias = self.rel(rel_bucket(pos[None, :] - pos[:, None])).permute(2, 0, 1).unsqueeze(0)   # (1, H, U, U)
        bias = bias + (~umask).float()[:, None, None, :] * -1e4
        for layer in self.layers:
            x = layer(x, bias)
        return self.norm(x) * umask.unsqueeze(2).float()


ENCODERS = {"glu": GLUEncoder, "lstm": LSTMEncoder, "transformer": TransformerEncoder}


def slot_span_mask(tags, labels, p):
    """Training-time slot-span masking (a span-level word dropout): each gold slot span is chosen with probability p,
    and all its tokens are flagged. Returns a (B, T) bool mask over tokens. Inputs are hidden, never added."""
    is_b = labels.tag_is_b.to(tags.device)[tags]
    in_slot = tags != 0                                                  # tag id 0 is "O"
    span_id = torch.cumsum(is_b.long(), dim=1)                           # gold BIO is well formed: I- follows B-/I-
    draw = torch.rand(tags.size(0), int(span_id.max()) + 1, device=tags.device) < p
    return in_slot & draw.gather(1, span_id)


def random_span_mask(lengths, T, rate, device, max_len=3):
    """Label-agnostic span dropout: random spans of 1..max_len tokens, placed anywhere, hiding about `rate` of the
    tokens. Unlike slot_span_mask it never looks at the labels, so <unk> carries no information about slots."""
    B = lengths.size(0)
    starts = torch.rand(B, T, device=device) < rate / ((max_len + 1) / 2)        # expected span length (1+max)/2
    span_len = torch.randint(1, max_len + 1, (B, T), device=device)
    pos = torch.arange(T, device=device)
    hide = torch.zeros(B, T, dtype=torch.bool, device=device)
    for k in range(max_len):                                                      # cover start .. start+len-1
        shifted = torch.zeros_like(hide)
        shifted[:, k:] = (starts & (span_len > k))[:, :T - k]
        hide |= shifted
    return hide & (pos.unsqueeze(0) < lengths.to(device).unsqueeze(1))


def masked_mean_max(h, mask):
    m = mask.unsqueeze(2).float()
    mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
    mx = h.masked_fill(m == 0, -1e4).max(1).values
    return torch.cat([mean, mx], dim=1)


class JointNet(nn.Module):
    """One ensemble member: shared input layer -> encoder -> intent (+ scenario) head and a slot head ('crf' or 'span').
    Both tasks are predicted by this single network."""

    def __init__(self, cfg, feats, labels, frozen):
        super().__init__()
        self.cfg, self.labels, C = cfg, labels, cfg["channels"]
        n_types = len(labels.slot_types)
        self.frozen = frozen
        self.task_emb = nn.Embedding(len(feats["units"]), cfg["task_dim"], padding_idx=0)
        self.char_cnn = CharCNN(len(feats["chars"]))
        self.ngram = nn.Embedding(feats["ngram_buckets"] + 1, cfg["ngram_dim"], padding_idx=0)
        self.lex = cfg["slot_lex"]
        in_dim = frozen.embedding_dim + cfg["task_dim"] + self.char_cnn.out_dim + cfg["ngram_dim"]
        if self.lex:
            self.lex_in = nn.Linear(2 * n_types, cfg["lex_dim"])
            in_dim += cfg["lex_dim"]
        self.in_drop = nn.Dropout(cfg["dropout"])
        self.proj = nn.Linear(in_dim, C)
        self.encoder = ENCODERS[cfg["encoder"]](cfg)
        self.drop = nn.Dropout(cfg["dropout"])
        ilex_dim = len(labels.intents) + 1 if cfg["intent_lex"] else 0
        self.intent_hidden = nn.Sequential(nn.Linear(2 * C + ilex_dim, C), nn.ReLU(), nn.Dropout(cfg["dropout"]))
        self.intent_out = nn.Linear(C, len(labels.intents))
        self.scen_out = nn.Linear(C, len(labels.scenarios))
        self.intent_label_emb = nn.Embedding(len(labels.intents), C)
        if self.lex:                                  # second read of the slot lexicon, right before the slot head
            self.lex_gate = nn.Linear(C + 2 * n_types, C)
            self.lex_val = nn.Linear(2 * n_types, C)
        if cfg["head"] == "crf":
            self.slot_head = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                                           nn.Linear(C, len(labels.tags)))
            self.crf = CRF(len(labels.tags), bio_constraints(labels.tags.itos))
        else:
            self.width_emb = nn.Embedding(MAX_SPAN, cfg["width_dim"])
            span_in = 3 * C + cfg["width_dim"] + (n_types if self.lex else 0)
            self.span_head = nn.Sequential(nn.Linear(span_in, C), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                                           nn.Linear(C, n_types + 1))
            self.slot_to_intent = nn.Linear(n_types, C)
        self.register_buffer("drop_p", feats["drop_p"])
        self.register_buffer("i2s", labels.i2s.clone())

    # -- shared trunk
    def trunk(self, batch):
        ids, umask = batch["units"], batch["units"] != 0
        m = umask.unsqueeze(2).float()
        tids, fids = ids, batch["frozen"]
        lex_on = torch.ones(ids.size(0), 1, 1, device=ids.device)
        if self.training:
            tids = ids.masked_fill((torch.rand(ids.shape, device=ids.device) < self.drop_p[ids]) & umask, 1)
            lex_on = (torch.rand(ids.size(0), 1, 1, device=ids.device) >= self.cfg["lex_drop"]).float()
            if self.cfg.get("span_mask_p", 0.0) > 0:
                hide = slot_span_mask(batch["tags"], self.labels, self.cfg["span_mask_p"])      # (B, T) tokens
                hide = hide.gather(1, batch["owner"]) & umask                                     # -> (B, U) units
                tids, fids = tids.masked_fill(hide, 1), fids.masked_fill(hide, 1)
            if self.cfg.get("rand_span_mask", 0.0) > 0:
                hide = random_span_mask(batch["lengths"], batch["tags"].size(1), self.cfg["rand_span_mask"],
                                        ids.device)
                hide = hide.gather(1, batch["owner"]) & umask
                tids, fids = tids.masked_fill(hide, 1), fids.masked_fill(hide, 1)
        g = batch["grams"]
        parts = [self.frozen(fids), self.task_emb(tids), self.char_cnn(batch["chars"]),
                 self.ngram(g).sum(2) / (g != 0).sum(2, keepdim=True).clamp(min=1)]
        tok_lex = batch["tok_lex"] * lex_on
        if self.lex:
            unit_lex = tok_lex.gather(1, batch["owner"].unsqueeze(2).expand(-1, -1, tok_lex.size(2)))
            parts.append(self.lex_in(unit_lex))
        x = self.proj(self.in_drop(torch.cat(parts, dim=2))) * m
        x = self.encoder(x, umask)
        pooled = masked_mean_max(x, umask)
        if self.cfg["intent_lex"]:
            pooled = torch.cat([pooled, batch["ilex"] * lex_on[:, :, 0]], dim=1)
        T = int(batch["lengths"].max())
        h = x.gather(1, batch["first"][:, :T].unsqueeze(2).expand(-1, -1, x.size(2)))
        tmask = torch.arange(T, device=x.device).unsqueeze(0) < batch["lengths"].to(x.device).unsqueeze(1)
        return h, tmask, pooled, tok_lex[:, :T], lex_on

    def compute(self, batch):
        h, tmask, pooled, tok_lex, lex_on = self.trunk(batch)
        z = self.intent_hidden(pooled)
        logits0 = self.intent_out(z)
        scen = self.scen_out(z)
        h = h + (F.softmax(logits0, 1) @ self.intent_label_emb.weight).unsqueeze(1)
        if self.lex:
            gate = torch.sigmoid(self.lex_gate(torch.cat([h, tok_lex], 2)))
            h = h + gate * self.lex_val(tok_lex)
        out = {"mask": tmask, "scen_logits": scen, "logits0": logits0}
        if self.cfg["head"] == "crf":
            out["emissions"] = self.slot_head(self.drop(h))
            logits = logits0
        else:
            B, T, C = h.shape
            i = torch.arange(T, device=h.device).repeat_interleave(MAX_SPAN)
            w = torch.arange(MAX_SPAN, device=h.device).repeat(T)
            j = (i + w).clamp(max=T - 1)
            valid = (i + w < T).unsqueeze(0) & tmask.gather(1, j.unsqueeze(0).expand(B, -1))
            csum = torch.cat([torch.zeros(B, 1, C, device=h.device), h.cumsum(1)], dim=1)
            mean = (csum[:, j + 1] - csum[:, i]) / (w + 1).unsqueeze(0).unsqueeze(2).float()
            rep = [h[:, i], h[:, j], mean, self.width_emb(w).unsqueeze(0).expand(B, -1, -1)]
            if self.lex:
                rep.append(batch["span_lex"][:, :T * MAX_SPAN] * lex_on)
            span_logits = self.span_head(self.drop(torch.cat(rep, dim=2)))
            out["span_logits"], out["span_valid"] = span_logits, valid
            probs = F.softmax(span_logits, 2)[:, :, 1:] * valid.unsqueeze(2).float()
            logits = self.intent_out(self.drop(z + self.slot_to_intent(torch.log1p(probs.sum(1)))))
        # hierarchical intent: log p(intent) + log p(its scenario)
        out["intent_logits"] = F.log_softmax(logits, 1) + F.log_softmax(scen, 1)[:, self.i2s]
        return out

    def loss(self, batch, out):
        cfg = self.cfg
        ls = cfg["label_smoothing"]
        loss = F.cross_entropy(out["intent_logits"], batch["intent"], label_smoothing=ls)
        loss = loss + cfg["scen_weight"] * F.cross_entropy(out["scen_logits"], batch["scenario"])
        T = out["mask"].size(1)
        if cfg["head"] == "crf":
            slot = self.crf.nll(out["emissions"], batch["tags"][:, :T], out["mask"]).mean()
        else:
            loss = loss + 0.5 * F.cross_entropy(out["logits0"], batch["intent"], label_smoothing=ls)
            grid = gold_span_grid(batch["tags"], batch["lengths"], self.labels)
            grid = grid[:, :out["span_logits"].size(1)].to(out["span_logits"].device)
            ce = F.cross_entropy(out["span_logits"].transpose(1, 2), grid, reduction="none")
            slot = (ce * out["span_valid"].float()).sum() / ce.size(0)
        return loss + cfg["slot_weight"] * slot

    @torch.no_grad()
    def predict_spans(self, batch, out):
        """Intent probabilities (B, K) and, per utterance, a dict {(start, width, type_id): confidence}."""
        probs = F.softmax(out["intent_logits"], 1)
        if self.cfg["head"] == "crf":
            paths, _ = crf_viterbi(out["emissions"], out["mask"], *self.crf.potentials())
            spans = [paths_to_spans(p, self.labels) for p in paths]
        else:
            spans = decode_span_logits(out["span_logits"], out["span_valid"], batch["lengths"])
        return probs, spans


def paths_to_spans(path, labels):
    """BIO tag ids -> {(start, width, type_id): 1.0}, with the scorer's rule for stray I- tags."""
    spans, start, cur = {}, None, 0
    for t, tid in enumerate(list(path) + [0]):
        typ = int(labels.tag_type[tid])
        if cur and (typ != cur or labels.tag_is_b[tid]):
            spans[(start, t - 1 - start, cur - 1)] = 1.0
            cur = 0
        if typ and not cur:
            start, cur = t, typ
    return spans


def decode_span_logits(span_logits, valid, lengths):
    logp = F.log_softmax(span_logits.float(), 2).cpu()
    best_type = logp[:, :, 1:].argmax(2) + 1
    gain = (logp.gather(2, best_type.unsqueeze(2)).squeeze(2) - logp[:, :, 0]).masked_fill(~valid.cpu(), -1.0)
    out = []
    for b, L in enumerate(lengths.tolist()):
        g = gain[b].view(-1, MAX_SPAN)[:L].tolist()
        chosen, _ = best_spans(g)
        out.append({(i, w, int(best_type[b, i * MAX_SPAN + w]) - 1): 1.0 for i, w in chosen})
    return out


def spans_to_text(span_dict, tokens, labels):
    return {(labels.slot_types[k], " ".join(tokens[i:i + w + 1])) for (i, w, k) in span_dict}


# ----------------------------------------------------------------------------------------------- training
def count_params(model):
    params = list(model.parameters())
    return sum(p.numel() for p in params), sum(p.numel() for p in params if p.requires_grad)


@torch.no_grad()
def member_outputs(model, batches):
    """Intent probabilities (N, K) and span dicts, in dataset order; plus average loss."""
    model.eval()
    probs, spans, tot, n = [], [], 0.0, 0
    for batch in batches:
        batch = to_device(batch)
        out = model.compute(batch)
        tot += model.loss(batch, out).item() * len(batch["lengths"])
        n += len(batch["lengths"])
        p, s = model.predict_spans(batch, out)
        probs.append(p.cpu())
        spans += s
    return torch.cat(probs), spans, tot / n


def evaluate_outputs(probs, spans, examples, labels):
    intents = [labels.intents.itos[i] for i in probs.argmax(1).tolist()]
    text = [spans_to_text(s, e["tokens"], labels) for s, e in zip(spans, examples)]
    return score(intents, text, examples)


def evaluate(model, batches, examples, labels):
    probs, spans, loss = member_outputs(model, batches)
    return {"loss": loss, **evaluate_outputs(probs, spans, examples, labels)}


def fit(model, cfg, train_enc, dev_batches, dev_examples, labels, ckpt_path, log=print):
    """Train on train only; after every epoch score dev and keep the best dev-combined checkpoint (raw or SWA)."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(train_enc) / cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["max_epochs"]
    warm = steps_per_epoch * cfg.get("warmup_epochs", 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    swa, swa_n = None, 0
    history, best = [], {"combined": -1.0}
    t_start = time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0
        for batch in make_batches(train_enc, cfg["batch_size"], shuffle=True, seed=cfg["seed"] * 1000 + epoch):
            batch = to_device(batch)
            loss = model.loss(batch, model.compute(batch))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg["clip"])
            opt.step()
            sched.step()
            running += loss.item() * len(batch["lengths"])
            seen += len(batch["lengths"])
        dev = evaluate(model, dev_batches, dev_examples, labels)
        candidates = [("raw", dev, None)]
        if cfg.get("swa_start") and epoch >= cfg["swa_start"]:
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if swa is None:
                swa = {k: v.clone() for k, v in state.items()}
            else:
                for k, v in state.items():
                    if v.dtype.is_floating_point:
                        swa[k].mul_(swa_n / (swa_n + 1)).add_(v, alpha=1 / (swa_n + 1))
            swa_n += 1
            model.load_state_dict(swa)
            candidates.append(("swa", evaluate(model, dev_batches, dev_examples, labels), copy.deepcopy(swa)))
            model.load_state_dict(state)
        row = {"epoch": epoch, "train_loss": running / seen, "dev_loss": dev["loss"], "dev_intent_acc": dev["intent_acc"],
               "dev_slot_f1": dev["slot_f1"], "dev_combined": dev["combined"], "lr": opt.param_groups[0]["lr"],
               "epoch_time_s": time.time() - t0}
        if len(candidates) > 1:
            s = candidates[1][1]
            row.update({"swa_intent_acc": s["intent_acc"], "swa_slot_f1": s["slot_f1"], "swa_combined": s["combined"]})
        history.append(row)
        improved = ""
        for kind, res, state in candidates:
            if res["combined"] > best["combined"]:
                best = {**res, "epoch": epoch, "kind": kind}
                torch.save({"state": state if state is not None else model.state_dict(), "epoch": epoch, "kind": kind,
                            "dev": res, "config": cfg}, ckpt_path)
                improved = f"  <- saved ({kind})"
        extra = f" | swa {row['swa_combined']:.2f}" if "swa_combined" in row else ""
        log(f"epoch {epoch:2d} | train {row['train_loss']:.4f} | dev loss {row['dev_loss']:.4f} | intent "
            f"{row['dev_intent_acc']:.2f} | slot F1 {row['dev_slot_f1']:.2f} | comb {row['dev_combined']:.2f}{extra} | "
            f"{row['epoch_time_s']:.1f}s{improved}")
    best["train_time_s"] = time.time() - t_start
    return history, best


# ----------------------------------------------------------------------------------------------- configs
BASE = dict(encoder="glu", head="crf", channels=192, dilations=[1, 2, 4, 8, 1, 2], dropout=0.4, task_dim=100,
            ngram_dim=50, width_dim=32, slot_lex=True, intent_lex=True, lex_dim=64, lex_drop=0.2,
            scen_weight=0.5, label_smoothing=0.1, slot_weight=1.0, lr=1e-3, weight_decay=0.01, batch_size=32,
            max_epochs=40, warmup_epochs=1, swa_start=30, clip=5.0)

ARCHS = {
    "glu_crf": {},
    "glu_span": {"head": "span"},
    "lstm_crf": {"encoder": "lstm", "channels": 256},
    "tf_crf": {"encoder": "transformer", "channels": 192, "layers": 4, "heads": 4, "ffn": 768, "dropout": 0.3,
               "lr": 5e-4},
}


def make_cfg(arch, seed, **over):
    return {**BASE, **ARCHS[arch], **over, "arch": arch, "seed": seed}


def make_frozen(table):
    return nn.Embedding.from_pretrained(table.clone(), freeze=True, padding_idx=0)


# ----------------------------------------------------------------------------------------------- ensemble
class MixedEnsemble:
    """Intent: weighted mean of member probabilities. Slots: every member votes (weight w_m) for each span it decoded;
    spans whose vote share reaches `threshold` are kept, highest vote first, skipping overlaps."""

    def __init__(self, weights=None, threshold=0.5):
        self.weights, self.threshold = weights, threshold

    def combine(self, member_probs, member_spans):
        M = len(member_probs)
        w = torch.tensor(self.weights if self.weights else [1.0] * M, dtype=torch.float)
        w = w / w.sum()
        probs = sum(wi * p for wi, p in zip(w, member_probs))
        spans = []
        for n in range(len(member_spans[0])):
            votes = Counter()
            for wi, ms in zip(w.tolist(), member_spans):
                for key in ms[n]:
                    votes[key] += wi
            chosen, used = {}, set()
            for (i, wd, k), v in sorted(votes.items(), key=lambda kv: (-kv[1], -kv[0][1], kv[0])):
                if v + 1e-9 < self.threshold:
                    break
                cover = set(range(i, i + wd + 1))
                if cover & used:
                    continue
                chosen[(i, wd, k)] = v
                used |= cover
            spans.append(chosen)
        return probs, spans
