#!/usr/bin/env python3
"""Run MACE training or model export, depending on the config you pass.

    python run_mace.py train.yaml        # stage 07: train
    python run_mace.py compile.yaml      # stage 08: export a checkpoint
"""

from mace.cli.run_train import main as mace_train_main
import sys


def run_mace(config_file):
    sys.argv = ["program", "--config", config_file]
    mace_train_main()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <config.yaml>")
    run_mace(sys.argv[1])
