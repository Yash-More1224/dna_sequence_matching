"""
Make the kmp package executable.

This allows running the CLI directly with: python -m kmp
"""

from .cli import main
import sys

if __name__ == '__main__':
    sys.exit(main())
