"""Verify and plot the four-column rotation example using SCS and direct ADMM."""
import argparse
from pathlib import Path

from isometry_pursuit.rotation_sweep import generate_rotation_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('figures/rotation_sweep'))
    args = parser.parse_args()
    result = generate_rotation_sweep(args.output_dir)
    print({k: result[k] for k in ('max_residual', 'max_objective_error')})


if __name__ == '__main__':
    main()
