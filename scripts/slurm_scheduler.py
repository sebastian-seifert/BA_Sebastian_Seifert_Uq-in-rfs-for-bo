#!/usr/bin/env python3
"""Submit a task manifest to Slurm in array chunks while respecting queue limits."""

from dyrf_bo.sweeps.slurm_scheduler import main

if __name__ == "__main__":
    main()
