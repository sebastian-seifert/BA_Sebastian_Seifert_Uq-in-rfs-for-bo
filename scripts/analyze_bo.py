#!/usr/bin/env python3
"""Gather, merge and analyse BO sweep results (scorecards and CARP-S reports)."""

import sys

from dyrf_bo.analysis.bo_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
