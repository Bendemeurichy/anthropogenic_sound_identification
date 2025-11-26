#!/usr/bin/env python3
"""
Command-line interface for YAMNet airplane detection finetuning.

This script provides a CLI for training the airplane detection classifier
on top of YAMNet features.

Example usage:
    python -m src.finetuning.cli \\
        --path-yamnet /path/to/yamnet_planes \\
        --path-data-train /path/to/training_data \\
        --path-data-csv /path/to/audio_paths.csv \\
        --path-save /path/to/save_models \\
        --epochs 1000 \\
        --learning-rate 0.001
"""

import argparse
import sys

from .config import FinetuneConfig
from .trainer import train


def parse_args(args: list | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Finetune YAMNet for airplane sound detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required paths
    parser.add_argument(
        "--path-yamnet",
        type=str,
        required=True,
        help="Path to yamnet_planes directory containing model files",
    )
    parser.add_argument(
        "--path-data-train",
        type=str,
        required=True,
        help="Path to directory containing training audio data",
    )
    parser.add_argument(
        "--path-save",
        type=str,
        required=True,
        help="Path to save trained models and metadata",
    )

    # Optional paths
    parser.add_argument(
        "--path-data-csv",
        type=str,
        default="",
        help="Path to CSV file with audio paths and labels",
    )

    # Classes
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["not_plane", "plane"],
        help="List of class names",
    )

    # Audio parameters
    parser.add_argument(
        "--patch-hop-seconds",
        type=float,
        default=0.096,
        help="Time hop between patches in seconds",
    )
    parser.add_argument(
        "--min-sample-seconds",
        type=float,
        default=1.0,
        help="Minimum audio duration in seconds",
    )
    parser.add_argument(
        "--max-sample-seconds",
        type=float,
        default=1000.0,
        help="Maximum audio duration in seconds",
    )

    # Augmentation
    parser.add_argument(
        "--num-augmentations",
        type=int,
        nargs="+",
        default=[0, 0],
        help="Number of augmentations per class",
    )

    # Model architecture
    parser.add_argument(
        "--num-hidden",
        type=int,
        nargs="+",
        default=[1024],
        help="List of hidden layer sizes",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes",
    )

    # Optimizer settings
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=1e-6,
        help="Learning rate decay for SGD optimizer",
    )
    parser.add_argument(
        "--no-nesterov",
        action="store_true",
        help="Disable Nesterov momentum for SGD",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--feature-extraction-method",
        type=int,
        default=1,
        choices=[0, 1],
        help="Feature extraction method (0: in-memory, 1: save to disk)",
    )

    return parser.parse_args(args)


def main(args: list | None = None) -> int:
    """Main entry point for the finetuning CLI.

    Args:
        args: Command line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parsed_args = parse_args(args)

    config = FinetuneConfig(
        path_yamnet=parsed_args.path_yamnet,
        path_data_train=parsed_args.path_data_train,
        path_data_csv=parsed_args.path_data_csv,
        path_save=parsed_args.path_save,
        classes=parsed_args.classes,
        patch_hop_seconds=parsed_args.patch_hop_seconds,
        min_sample_seconds=parsed_args.min_sample_seconds,
        max_sample_seconds=parsed_args.max_sample_seconds,
        num_augmentations=parsed_args.num_augmentations,
        num_hidden=parsed_args.num_hidden,
        num_classes=parsed_args.num_classes,
        optimizer_type=parsed_args.optimizer,
        learning_rate=parsed_args.learning_rate,
        momentum=parsed_args.momentum,
        decay=parsed_args.decay,
        nesterov=not parsed_args.no_nesterov,
        epochs=parsed_args.epochs,
        validation_split=parsed_args.validation_split,
        feature_extraction_method=parsed_args.feature_extraction_method,
    )

    try:
        result = train(config)
        print(f"\nModel saved to: {result['path_model']}")
        return 0
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
