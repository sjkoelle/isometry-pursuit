"""Generate the supplemental real-data optimizer support-stability figure."""

import argparse

from isometry_pursuit.optimizer_stability import plot_optimizer_support_stability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-dir', default='results/cloud/zero-start-smoke-001')
    parser.add_argument('--extended-dir', default='results/cloud/large-zero-smoke-001')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--output-dir', default='figures')
    args = parser.parse_args()
    summary = plot_optimizer_support_stability(args.smoke_dir, args.extended_dir, args.results_dir, args.output_dir)
    for dataset, result in summary.items():
        print(f"{dataset}: {result['initial_matches']}/10 initial and {result['latest_matches']}/10 latest supports match the reported support")


if __name__ == '__main__':
    main()
