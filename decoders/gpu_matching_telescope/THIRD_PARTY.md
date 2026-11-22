# Third-party matching source

This package uses the installed PyMatching 2.4.0 package for CPU correlated
decoding and full-candidate scoring. Conditioned solves use the bundled
PyMatching sparse-blossom core through the local native adapter. The legacy CUDA
matcher adapts those translation units at runtime through `correlated_source.py`
and `native/correlated_matching.cu`.

The matching core is copied unmodified from
[PyMatching v2.4.0](https://github.com/oscarhiggott/PyMatching/tree/v2.4.0),
copyright 2022 PyMatching Contributors, under Apache-2.0. The complete license and
37 upstream Git blob identities are retained in
`src/gpu_matching_telescope/native/vendor/pymatching_2_4_0/`.

The local `native/matching.cc` adapter includes adapted timeline, extraction and
negative-edge cancellation logic from that release's `mwpm_decoding.cc`; its
header records the modifications. It reuses the upstream matching algorithm.

The restored Windows support and matching binaries retain their original bytes.
Their source and binary SHA-256 identities are recorded in
`native/prebuilt/windows/manifest.json`. Native source, vendor files, licenses,
and the manifest were copied without modification.

This package has its own Python namespace and native build location.
