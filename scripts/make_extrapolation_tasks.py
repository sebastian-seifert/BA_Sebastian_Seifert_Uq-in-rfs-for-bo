#!/usr/bin/env python3
"""Write the task manifest for the extrapolation UQ sweep."""

import sys

from dyrf_bo.sweeps.extrapolation_tasks import main

if __name__ == "__main__":
    sys.exit(main())
