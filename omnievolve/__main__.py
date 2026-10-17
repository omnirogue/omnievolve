# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""`python -m omnievolve` runs the command-line interface."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
