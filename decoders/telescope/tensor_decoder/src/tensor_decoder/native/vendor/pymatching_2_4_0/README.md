# Pinned PyMatching matching core

This directory contains unmodified matching-core sources from PyMatching v2.4.0,
https://github.com/oscarhiggott/PyMatching/tree/v2.4.0, under Apache-2.0 (LICENSE).
ORIGIN.json records upstream Git blob hashes. `latency_native.audit_vendor()`
verifies all upstream bytes. The manifest's tree SHA identifies the upstream
source tree, not a commit.

Only the non-correlated sparse-blossom core and its headers are included; tests,
the Python binding, Stim integration, and the UserGraph driver are excluded.
The adjacent `latency_kernel.cc` is our separate fixed-topology adapter. Its
timeline/extraction code is adapted from the same release's mwpm_decoding.cc.
PyMatching performs the matching solves; this project does not claim a new
blossom algorithm. The initial correlation-aware solve continues to use the
installed PyMatching Python package.
