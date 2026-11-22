"""Adapt the pinned sparse-blossom translation units for NVRTC."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from ._build import VENDOR
from .matching import audit_vendor

_CORE = "pymatching/sparse_blossom/"
_DEVICE_INTEGER_TYPES = rb"""
using uint8_t = unsigned char;
using int8_t = signed char;
using uint32_t = unsigned int;
using int32_t = int;
using uint64_t = unsigned long long;
using int64_t = long long;
constexpr size_t SIZE_MAX = static_cast<size_t>(-1);
"""
_GPU_GRAPHS = {
    _CORE + "flooder/graph.h": (
        "MatchingGraph",
        """class MatchingGraph {
   public:
    std::vector<DetectorNode> nodes;
    pm::total_weight_int negative_weight_sum;
    size_t num_nodes;
    size_t num_observables;
    double normalising_constant;

    MatchingGraph();
    MatchingGraph(size_t num_nodes, size_t num_observables);
    MatchingGraph(size_t num_nodes, size_t num_observables, double normalising_constant);
    MatchingGraph(MatchingGraph&& graph) noexcept;
};
""",
    ),
    _CORE + "search/search_graph.h": (
        "SearchGraph",
        """class SearchGraph {
   public:
    std::vector<SearchDetectorNode> nodes;
    size_t num_nodes;
    std::vector<std::pair<size_t, size_t>> negative_weight_edges;

    SearchGraph();
    explicit SearchGraph(size_t num_nodes);
    SearchGraph(SearchGraph&& graph) noexcept;
};
""",
    ),
}

# No changes to the matching timeline, search, or radix queue ordering.
_UNUSED_METHODS = (
    "sync_negative_weight_observables_and_detection_events",
    "compute_stitch_radius_at_time_bounded_by_region_towards_neighbor",
    "extract_paths_from_match_edges",
    "all_nodes_in_tree",
    "to_vector",
)
_METHOD = re.compile(
    r"(?m)^[ \t]*(?:template[^\n]*\n[ \t]*)?"
    r"(?:(?:inline |friend )*std::(?:string .*?\bstr|ostream\s*&\s*(?:[\w:]+::)?operator<<)"
    rf"|[^;{{}}\n]*\b(?:{'|'.join(_UNUSED_METHODS)}))\([^;{{}}]*[;{{]"
)
_BODY_TOKEN = re.compile(r'''//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[{}]''', re.S)
_HEADER = re.compile(r'^#include[ \t]+(?:<([^>]+)>|"([^"]+)")[^\n]*\n', re.M)


def _closing_brace(source: str, opening: int) -> int:
    depth = 0
    for token in _BODY_TOKEN.finditer(source, opening):
        if token.group() == "{":
            depth += 1
        elif token.group() == "}":
            depth -= 1
            if depth == 0:
                return token.end()
    raise RuntimeError("unclosed function body in pinned CUDA matching source")


def _without_host_methods(source: str) -> str:
    for match in reversed(list(_METHOD.finditer(source))):
        end = match.end()
        if source[end - 1] == "{":
            end = _closing_brace(source, end - 1)
        if source[end : end + 1] == "\n":
            end += 1
        source = source[: match.start()] + source[end:]
    return source


def _device_source(name: str, source: str) -> str:
    def replace_include(match: re.Match) -> str:
        quoted = match[2]
        if quoted is None:
            return ""
        if "/" not in quoted:
            quoted = (PurePosixPath(name).parent / quoted).as_posix()
        return f'#include "{quoted}"\n'

    source = _HEADER.sub(replace_include, source)
    if name in _GPU_GRAPHS:
        graph, replacement = _GPU_GRAPHS[name]
        first = "struct PreviousWeight" if graph == "MatchingGraph" else f"class {graph}"
        left = source.index(first)
        right = source.index("}  // namespace pm")
        source = source[:left] + replacement + "\n" + source[right:]
    source = _without_host_methods(source)
    source = re.sub(
        r"(?m)^[ \t]*std::vector<std::vector<ImpliedWeight>> neighbor_implied_weights;\n",
        "",
        source,
    )
    source = re.sub(
        r"(?m)^[ \t]*std::vector<std::vector<size_t>>\s+neighbor_observable_indices;[^\n]*\n",
        "",
        source,
    )
    source = re.sub(r"(?m)^[ \t]*(?:std::stringstream ss;|ss <<)[^\n]*\n", "", source)
    source = re.sub(r"\bthrow std::invalid_argument\([^;]*\);", "gpu::fail();", source)
    source = re.sub(r"\bassert\(([^;]*)\);", r"if (!(\1)) gpu::fail();", source)
    source = re.sub(r"\bmalloc\(", "gpu::allocate(", source)
    source = re.sub(r"\bfree\(", "gpu::deallocate(", source)
    source = source.replace(
        "std::vector<SearchGraphEdge> path_edges_2 = {collision_edge};",
        "std::vector<SearchGraphEdge> path_edges_2;\n    path_edges_2.push_back(collision_edge);",
    )
    return source.replace("std::", "gpu::")


def vendor_sources() -> tuple[bytes, dict[bytes, bytes]]:
    """Return the generated source and its virtual NVRTC headers."""
    audit_vendor()
    root = VENDOR / "src"
    replacements = {PurePosixPath(name).with_suffix(".cc").as_posix() for name in _GPU_GRAPHS}
    headers = {}
    translation_units = []
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.suffix not in (".h", ".inl", ".cc") or name in replacements:
            continue
        headers[name.encode()] = _device_source(name, path.read_text(encoding="utf-8")).encode()
        if path.suffix == ".cc":
            translation_units.append(f'#include "{name}"\n')
    return _DEVICE_INTEGER_TYPES + "".join(translation_units).encode(), headers
