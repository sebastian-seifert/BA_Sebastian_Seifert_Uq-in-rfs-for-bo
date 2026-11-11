#!/usr/bin/env python3
"""Run one extrapolation UQ experiment (dimension, n_train, function, surrogate, seed)."""

import sys

from dyrf_bo.extrapolation.cli import main

if __name__ == "__main__":
    sys.exit(main())
