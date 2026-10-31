#!/usr/bin/env python3
"""Public entry point for the multi-model experiment suite."""
from lab.runner import main
import sys

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped. Completed stages and checkpoints are preserved; rerun the command to resume.', file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
