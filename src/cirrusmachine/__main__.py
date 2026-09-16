"""Small CPU synthesis check: python -m cirrusmachine --help."""

from pathlib import Path
import argparse

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce

from .synthesis import MockSynthesisConfig, synthesize_single_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional 2D primary-HDU FITS image")
    parser.add_argument("--output", type=Path, default=Path("result/smoke.fits"))
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4703)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if args.size < 16 or args.size % 2 or args.steps < 1:
        parser.error("size must be even and at least 16; steps must be positive")
    import torch
    torch.set_num_threads(4)
    if args.device == "gpu" and not torch.cuda.is_available():
        parser.error("this smoke workflow requires CUDA for --device gpu")
    if args.input:
        image = np.asarray(fits.getdata(args.input, ext=0), dtype=np.float64)
        if image.ndim != 2 or min(image.shape) < args.size:
            parser.error("input must be 2D and at least size pixels on each axis")
        factor = min(image.shape) // args.size
        side = factor * args.size
        image = block_reduce(image[:side, :side], factor, func=np.mean)
    else:
        image = np.random.default_rng(args.seed).normal(size=(args.size, args.size))
    if not np.isfinite(image).all() or image.std() == 0:
        parser.error("input must have finite values and nonzero variance")
    result = synthesize_single_image(image, config=MockSynthesisConfig(
        estimator_name="s_mean_iso", J=3, L=4, seed=args.seed,
        device=args.device, steps=args.steps,
    ))
    output = np.asarray(result["image_out"])
    if not np.isfinite(output).all() or output.shape[-2:] != image.shape:
        raise RuntimeError("synthesis failed finite-value or output-shape checks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["COMMENT"] = "Installation smoke check; not a scientifically converged mock."
    fits.writeto(args.output, output.astype(np.float32), header, overwrite=False)
    print(f"CPU/GPU smoke check passed: {output.shape}, output={args.output}")


if __name__ == "__main__":
    main()
