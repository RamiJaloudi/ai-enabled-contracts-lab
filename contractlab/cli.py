from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


def command_doctor() -> int:
    import torch

    from contractlab import __version__
    from contractlab.modeling import device_summary

    report = {
        "contractlab": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        **device_summary(),
    }
    print(json.dumps(report, indent=2))
    if sys.version_info[:2] != (3, 12):
        print("WARNING: AI-Enabled Contracts Lab is tested with Python 3.12.", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contractlab",
        description="AI-Enabled Contracts Lab command-line tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Report Python, PyTorch, and GPU readiness")

    train = subparsers.add_parser("train", help="Run a configured training experiment")
    train.add_argument("--config", required=True, type=Path)

    cuad = subparsers.add_parser(
        "prepare-cuad", help="Download and convert positive CUAD answer spans"
    )
    cuad.add_argument("--output", required=True, type=Path)
    cuad.add_argument("--max-rows", type=int, default=None)

    notebook = subparsers.add_parser(
        "export-notebook", help="Export a configured experiment as a notebook"
    )
    notebook.add_argument("--config", required=True, type=Path)
    notebook.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "doctor":
        return command_doctor()
    if args.command == "train":
        from contractlab.config import TrainingConfig
        from contractlab.training import run_training

        config = TrainingConfig.load(args.config)
        results = run_training(config)
        print(json.dumps(results["test_metrics"], indent=2))
        return 0
    if args.command == "prepare-cuad":
        from contractlab.data import prepare_cuad

        output = prepare_cuad(args.output, args.max_rows)
        print(output)
        return 0
    if args.command == "export-notebook":
        from contractlab.codegen import write_notebook
        from contractlab.config import TrainingConfig

        config = TrainingConfig.load(args.config)
        output = args.output or args.config.with_suffix(".ipynb")
        print(write_notebook(config, output))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
