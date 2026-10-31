"""Train GloVe BiGRU slots with diffusion denoising and CRF likelihood."""

from pathlib import Path

from main_diffusion_glove import parse_args, run
from sc4002.models.joint_bigru_glove_diffusion_crf import JointBiGRUGloveDiffusionCRF


if __name__ == "__main__":
    args = parse_args()
    if args.checkpoint_dir == Path("checkpoints/diffusion_glove"):
        args.checkpoint_dir = Path("checkpoints/diffusion_crf_glove")
    if args.log_dir == Path("logs/diffusion_glove"):
        args.log_dir = Path("logs/diffusion_crf_glove")
    run(args, model_class=JointBiGRUGloveDiffusionCRF,
        model_type="glove_bigru_diffusion_crf")
