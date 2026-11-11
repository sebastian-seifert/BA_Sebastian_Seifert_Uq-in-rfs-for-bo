#!/usr/bin/env python3
"""Execute an extrapolation task file locally with a process pool."""

import sys

from dyrf_bo.sweeps.local_runner import main

if __name__ == "__main__":
    sys.exit(main())
