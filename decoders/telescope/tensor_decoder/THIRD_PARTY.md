# Third-party source

This attribution describes the retained reference engine. The main telescope
uses the original `../native/owned_*` implementation and does not
compile or load these third-party matching sources.

The sparse-blossom matching core is copied unmodified from
[PyMatching v2.4.0](https://github.com/oscarhiggott/PyMatching/tree/v2.4.0),
copyright 2022 PyMatching Contributors, under Apache-2.0. The full license and
per-file Git blob hashes are bundled in
`src/tensor_decoder/native/vendor/pymatching_2_4_0/`.

The local `native/matching.cc` adapter includes adapted timeline, extraction and
negative-edge cancellation logic from that release's `mwpm_decoding.cc`; its
header records the modifications. The reference adapter uses the upstream matching algorithm. Initial correlated decoding also uses the
installed PyMatching 2.4.0 package.

The validated Windows binaries were carried over without changing their bytes.
Their source and binary hashes are in `native/prebuilt/windows/manifest.json`.
The support binary includes unused historical exports; only the current path
and required small-code kernel sources are retained and used here.
