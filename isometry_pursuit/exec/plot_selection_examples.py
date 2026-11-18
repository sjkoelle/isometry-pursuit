"""Plot saved D=2 Ethanol and synthetic examples for the four search methods."""
import argparse
from pathlib import Path

from isometry_pursuit.selection_examples import generate_selection_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('figures/selection_examples'))
    args = parser.parse_args()
    result = generate_selection_examples(args.root, args.output_dir)
    for dataset in ('ethanol', 'synthetic'):
        print(dataset, [(row['case'], row['id']) for row in result[dataset]])


if __name__ == '__main__':
    main()
