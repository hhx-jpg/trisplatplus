# Vendored Depth-Anything-3 bridge

This directory contains the DA3 Python source needed by TriSplat++,
including the TSDPT and `TSAdapter` additions used by the DA3 geometry path.
The source is kept under the upstream DA3 license in `LICENSE`.

The encoder prefers this copy when `DA3_ROOT` is unset. For a separately
managed DA3 checkout, set `DA3_ROOT=/path/to/Depth-Anything-3` (or its
`src` directory) and `DA3_CHECKPOINT=/path/to/DA3-GIANT-1.1`.

Model checkpoints are not vendored here.
