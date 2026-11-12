"""
Official MASSIVE evaluation (intent accuracy, slot micro-F1, exact-match accuracy).

`convert_to_bio` and `eval_preds` are copied verbatim from the MASSIVE repository,
https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/src/massive/utils/training_utils.py
and the body of `annot_to_slot_labels` is the gold-label loop from
https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/scripts/create_hf_dataset.py
Only `bio_to_slot_labels` and `annot_to_slot_labels` (as a function) are ours. The original notice follows.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from math import sqrt

from seqeval.metrics import f1_score
import sklearn.metrics as sklm


# ----------------------------------------------------------------------------- our adapters
def annot_to_slot_labels(annot_utt):
    """Gold per-word slot labels ('Other' outside a slot), exactly as MASSIVE's create_hf_dataset.py builds them."""
    labels = []
    label = 'Other'
    split_annot_utt = annot_utt.split()
    idx = 0
    while idx < len(split_annot_utt):
        if split_annot_utt[idx].startswith('['):
            label = split_annot_utt[idx].lstrip('[')
            idx += 2
        elif split_annot_utt[idx].endswith(']'):
            labels.append(label)
            label = 'Other'
            idx += 1
        else:
            labels.append(label)
            idx += 1
    return labels


def bio_to_slot_labels(tags):
    """One BIO tag per word -> MASSIVE's per-word slot labels ('O' -> 'Other', 'B-x' / 'I-x' -> 'x')."""
    return ['Other' if t == 'O' else t[2:] for t in tags]


# ----------------------------------------------------------------------------- verbatim from MASSIVE
def convert_to_bio(seq_tags, outside='Other', labels_merge=None):
    """
    Converts a sequence of tags into BIO format. EX:

        ['city', 'city', 'Other', 'country', -100, 'Other']
        to
        ['B-city', 'I-city', 'O', 'B-country', 'I-country', 'O']
        where outside = 'Other' and labels_merge = [-100]

    :param seq_tags: the sequence of tags that should be converted
    :type seq_tags: list
    :param outside: The label(s) to put outside (ignore). Default: 'Other'
    :type outside: str or list
    :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
    :type labels_merge: str or list
    :return: a BIO-tagged sequence
    :rtype: list
    """

    seq_tags = [str(x) for x in seq_tags]

    outside = [outside] if type(outside) != list else outside
    outside = [str(x) for x in outside]

    if labels_merge:
        labels_merge = [labels_merge] if type(labels_merge) != list else labels_merge
        labels_merge = [str(x) for x in labels_merge]
    else:
        labels_merge = []

    bio_tagged = []
    prev_tag = None
    for tag in seq_tags:
        if prev_tag == None and tag in labels_merge:
            bio_tagged.append('O')
        elif tag in outside:
            bio_tagged.append('O')
            prev_tag = tag
        elif tag != prev_tag and tag not in labels_merge:
            bio_tagged.append('B-' + tag)
            prev_tag = tag
        elif tag == prev_tag or tag in labels_merge:
            if prev_tag in outside:
                bio_tagged.append('O')
            else:
                bio_tagged.append('I-' + prev_tag)

    return bio_tagged

def eval_preds(pred_intents=None, lab_intents=None, pred_slots=None, lab_slots=None,
               eval_metrics='all', labels_ignore='Other', labels_merge=None, pad='Other'):
    """
    Function to evaluate the predictions from a model

    :param pred_intents: a list of predicted intents
    :type pred_intents: list
    :param lab_intents: a list of intents labels (ground truth)
    :type lab_intents: list
    :param pred_slots: a list of predicted slots, where each entry is a list of token-based slots
    :type pred_slots: list
    :param lab_slots: a list of slots labels (ground truth)
    :type lab_slots: list
    :param eval_metrics: The metrics to include. Options are 'all', 'intent_acc', 'ex_match_acc',
                         'slot_micro_f1'
    :type eval_metrics: str
    :param labels_ignore: The labels to ignore (prune away). Default: ['Other']
    :type labels_ignore: str or list
    :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
    :type labels_merge: str or list
    :param pad: The value to use when padding slot predictions to match the length of ground truth
    :type pad: str
    """

    results = {}

    # Check lengths
    if pred_intents is not None and lab_intents is not None:
        assert len(pred_intents) == len(lab_intents),"pred_intents and lab_intents must be same len"
    if pred_slots is not None and lab_slots is not None:
        assert len(pred_slots) == len(lab_slots), "pred_slots and lab_slots must be same length"

    if ('intent_acc' in eval_metrics) or ('all' in eval_metrics):
        intent_acc = sklm.accuracy_score(lab_intents, pred_intents)
        results['intent_acc'] = intent_acc
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        results['intent_acc_stderr'] = sqrt(intent_acc*(1-intent_acc)/len(pred_intents))

    if lab_slots is not None and pred_slots is not None:
        bio_slot_labels, bio_slot_preds = [], []
        for lab, pred in zip(lab_slots, pred_slots):

            # Pad or truncate prediction as needed using `pad` arg
            if type(pred) == list:
                pred = pred[:len(lab)] + [pad]*(len(lab) - len(pred))

            # Fix for Issue 21 -- subwords after the first one from a word should be ignored
            for i, x in enumerate(lab):
                if x == -100:
                    pred[i] = -100

            # convert to BIO
            bio_slot_labels.append(
                convert_to_bio(lab, outside=labels_ignore, labels_merge=labels_merge)
            )
            bio_slot_preds.append(
                convert_to_bio(pred, outside=labels_ignore, labels_merge=labels_merge)
            )

    if ('slot_micro_f1' in eval_metrics) or ('all' in eval_metrics):

        # from seqeval
        smf1 = f1_score(bio_slot_labels, bio_slot_preds)
        results['slot_micro_f1'] = smf1
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        total_slots = sum([len(x) for x in bio_slot_preds])
        results['slot_micro_f1_stderr'] = sqrt(smf1*(1-smf1)/total_slots)

    if ('ex_match_acc' in eval_metrics) or ('all' in eval_metrics):
        # calculate exact match accuracy (~0.01 seconds)
        matches = 0
        denom = 0
        for p_int, p_slot, l_int, l_slot in zip(pred_intents,
                                                bio_slot_preds,
                                                lab_intents,
                                                bio_slot_labels):

            if (p_int == l_int) and (p_slot == l_slot):
                matches += 1
            denom += 1
        emacc = matches / denom

        results['ex_match_acc'] = emacc
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        results['ex_match_acc_stderr'] = sqrt(emacc*(1-emacc)/len(pred_intents))

    return results
