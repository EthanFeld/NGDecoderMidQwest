// Copyright 2022 PyMatching Contributors
// Licensed under the Apache License, Version 2.0; see vendor/pymatching_2_4_0/LICENSE.
// Modified: persistent topology, weight updates, correction maps and reusable workspaces.
// Exact matching with a persistent fixed topology.
// Timeline/extraction follows PyMatching 2.4.0's mwpm_decoding.cc (Apache-2.0).
// The unmodified matching core and its license are in vendor/pymatching_2_4_0.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "pymatching/sparse_blossom/matcher/mwpm.h"
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

static thread_local std::string last_error;
static constexpr double MAX_WEIGHT=16777215.0;
static uint64_t edge_key(int64_t a, int64_t b) {
    if (b>=0 && b<a) std::swap(a,b);
    return (uint64_t)(uint32_t)a<<32 | (uint32_t)(b+1);
}

struct FixedMatcher {
    int n,m,other_count;
    std::unique_ptr<pm::Mwpm> matcher;
    std::vector<int32_t> u,v;
    std::vector<uint8_t> logical,negative;
    std::vector<pm::weight_int*> forward,reverse;
    std::unordered_map<uint64_t,int32_t> correction_map;
    std::vector<int32_t> baseline;
    std::vector<uint64_t> events;
    std::vector<int> edge_order,search_forward,search_reverse;

    FixedMatcher(int nodes,int edges,const int32_t* a,const int32_t* b,
        const uint8_t* labels,int others,int maps,const uint64_t* keys,const int32_t* ids)
        :n(nodes),m(edges),other_count(others),u(a,a+edges),v(b,b+edges),
         logical(labels,labels+edges),negative(nodes,0),forward(edges),reverse(edges,nullptr) {
        pm::MatchingGraph graph(n,1,2.0);
        std::vector<int> boundaries;
        for (int j=0;j<m;++j) {
            if (u[j]<0 || u[j]>=n || v[j]>=n || u[j]==v[j])
                throw std::invalid_argument("invalid fixed graph edge");
            if (v[j]<0) boundaries.push_back(j);
            else graph.add_edge(u[j],v[j],2,logical[j] ? std::vector<size_t>{0} : std::vector<size_t>{});
        }
        // UserGraph converts ordinary edges in insertion order, then boundary
        // edges in node order. Preserve this order, including boundary slot 0.
        std::sort(boundaries.begin(),boundaries.end(),[&](int x,int y){return u[x]<u[y];});
        for (int j:boundaries)
            graph.add_boundary_edge(u[j],2,logical[j] ? std::vector<size_t>{0} : std::vector<size_t>{});
        for (int j=0;j<m;++j) if (v[j]>=0) edge_order.push_back(j);
        edge_order.insert(edge_order.end(),boundaries.begin(),boundaries.end());
        matcher=std::make_unique<pm::Mwpm>(pm::GraphFlooder(std::move(graph)));
        for (int j=0;j<m;++j) {
            auto& node=matcher->flooder.graph.nodes[u[j]];
            auto target=v[j]<0 ? nullptr : &matcher->flooder.graph.nodes[v[j]];
            forward[j]=&node.neighbor_weights[node.index_of_neighbor(target)];
            if (target) reverse[j]=&target->neighbor_weights[target->index_of_neighbor(&node)];
        }
        correction_map.reserve(maps*2);
        for (int j=0;j<maps;++j) {
            if (!correction_map.emplace(keys[j],ids[j]).second)
                throw std::invalid_argument("duplicate correction-map key");
        }
        events.reserve(n);
    }

    void ensure_search() {
        if (!search_forward.empty()) return;
        pm::SearchGraph graph(n);
        for (int j:edge_order) {
            std::vector<size_t> obs=logical[j] ? std::vector<size_t>{0} : std::vector<size_t>{};
            if (v[j]<0) graph.add_boundary_edge(u[j],2,obs);
            else graph.add_edge(u[j],v[j],2,obs);
        }
        matcher->search_flooder.~SearchFlooder();
        new (&matcher->search_flooder) pm::SearchFlooder(std::move(graph));
        search_forward.resize(m); search_reverse.resize(m,-1);
        auto& nodes=matcher->search_flooder.graph.nodes;
        for (int j=0;j<m;++j) {
            search_forward[j]=nodes[u[j]].index_of_neighbor(v[j]<0 ? nullptr : &nodes[v[j]]);
            if (v[j]>=0) search_reverse[j]=nodes[v[j]].index_of_neighbor(&nodes[u[j]]);
        }
    }

    void prepare(const int64_t* pairs,int count,uint8_t* parity) {
        std::fill(parity,parity+other_count+1,0);
        baseline.clear();
        for (int j=0;j<count;++j) {
            auto found=correction_map.find(edge_key(pairs[2*j],pairs[2*j+1]));
            if (found==correction_map.end()) throw std::invalid_argument("correction edge absent from model");
            int id=found->second;
            if (id>0) baseline.push_back(id-1);
            else parity[-id-1]^=1;
        }
    }

    void joint_events(const int64_t* pairs,int count,uint8_t* out) {
        prepare(pairs,count,out+m);
        std::fill(out,out+m,0);
        for (int id:baseline) out[id]^=1;
    }

    static void flip_edge(const pm::SearchGraphEdge& e) {
        e.detector_node->neighbor_markers[e.neighbor_index]^=pm::FLIPPED;
        auto other=e.detector_node->neighbors[e.neighbor_index];
        if (other) other->neighbor_markers[other->index_of_neighbor(e.detector_node)]^=pm::FLIPPED;
    }

    // Adapted from the pinned mwpm_decoding.cc. Preserve path and cancellation
    // order: different minimum-cost ties can condition a different next proposal.
    void extract_edges(std::vector<int64_t>& out) {
        auto& solver=*matcher;
        auto& flooder=solver.flooder;
        auto& search=solver.search_flooder;
        flooder.match_edges.clear();
        auto shatter=[&](const std::vector<uint64_t>& ds) {
            for (auto i:ds) if (flooder.graph.nodes[i].region_that_arrived)
                solver.shatter_blossom_and_extract_match_edges(
                    flooder.graph.nodes[i].region_that_arrived_top,flooder.match_edges);
        };
        shatter(events); shatter(flooder.negative_weight_detection_events);
        out.clear();
        for (const auto& pair:search.graph.negative_weight_edges) {
            auto a=&search.graph.nodes[pair.first];
            auto b=pair.second==SIZE_MAX ? nullptr : &search.graph.nodes[pair.second];
            flip_edge({a,a->index_of_neighbor(b)});
            out.push_back(pair.first); out.push_back(b ? (int64_t)pair.second : -1);
        }
        for (const auto& match:flooder.match_edges) {
            size_t a=match.loc_from-&flooder.graph.nodes[0];
            size_t b=match.loc_to ? match.loc_to-&flooder.graph.nodes[0] : SIZE_MAX;
            search.iter_edges_on_shortest_path_from_middle(a,b,[&](const pm::SearchGraphEdge& e) {
                flip_edge(e);
                auto other=e.detector_node->neighbors[e.neighbor_index];
                out.push_back(e.detector_node-&search.graph.nodes[0]);
                out.push_back(other ? other-&search.graph.nodes[0] : -1);
            });
        }
        for (size_t i=0;i<out.size()/2;) {
            auto& node=search.graph.nodes[out[2*i]];
            auto v=out[2*i+1];
            size_t k=node.index_of_neighbor(v<0 ? nullptr : &search.graph.nodes[v]);
            if (!(node.neighbor_markers[k]&pm::FLIPPED)) {
                out[2*i]=out[out.size()-2]; out[2*i+1]=out[out.size()-1];
                out.resize(out.size()-2);
            } else {flip_edge({&node,k});++i;}
        }
    }

    void canonical_empty_workspace() {
        auto& solver=*matcher;
        if (solver.node_arena.allocated.size()!=solver.node_arena.available.size() ||
            solver.flooder.region_arena.allocated.size()!=solver.flooder.region_arena.available.size())
            throw std::runtime_error("matching workspace not empty after previous decode");
        for (auto& node:solver.flooder.graph.nodes) node.reset();
        solver.flooder.queue.clear();
        solver.flooder.queue.cur_time=0;
        // Reuse allocations, but always hand them out in original allocation
        // order. Earlier shot complexity must not select a different tie break.
        solver.node_arena.available.clear();
        for (auto it=solver.node_arena.allocated.rbegin();it!=solver.node_arena.allocated.rend();++it)
            solver.node_arena.available.push_back(*it);
        solver.flooder.region_arena.available.clear();
        for (auto it=solver.flooder.region_arena.allocated.rbegin();it!=solver.flooder.region_arena.allocated.rend();++it)
            solver.flooder.region_arena.available.push_back(*it);
    }

    void solve(const double* weights,const uint8_t* syndrome,uint8_t* label,double* weight,double* gain,
        std::vector<int64_t>* out_edges=nullptr) {
        double maximum=0; bool integral=true;
        for (int j=0;j<m;++j) {
            if (!std::isfinite(weights[j])) throw std::invalid_argument("nonfinite matching weight");
            maximum=std::max(maximum,std::abs(weights[j]));
            integral&=std::round(weights[j])==weights[j];
        }
        if (maximum>MAX_WEIGHT) throw std::invalid_argument("matching weight exceeds PyMatching limit");
        double scale=integral ? 1.0 : MAX_WEIGHT/maximum;
        if (out_edges) ensure_search();
        canonical_empty_workspace();
        auto& solver=*matcher;
        auto& flooder=solver.flooder;
        flooder.negative_weight_detection_events.clear();
        flooder.negative_weight_obs_mask=0;
        flooder.negative_weight_sum=0;
        std::fill(negative.begin(),negative.end(),0);
        if (out_edges) matcher->search_flooder.graph.negative_weight_edges.clear();
        for (int j=0;j<m;++j) {
            auto w=(pm::signed_weight_int)std::round(weights[j]*scale)*2;
            *forward[j]=(pm::weight_int)std::abs(w);
            if (reverse[j]) *reverse[j]=(pm::weight_int)std::abs(w);
            if (out_edges) {
                auto& nodes=matcher->search_flooder.graph.nodes;
                nodes[u[j]].neighbor_weights[search_forward[j]]=std::abs(w);
                nodes[u[j]].neighbor_markers[search_forward[j]]=w<0 ? pm::WEIGHT_SIGN : 0;
                if (v[j]>=0) {
                    nodes[v[j]].neighbor_weights[search_reverse[j]]=std::abs(w);
                    nodes[v[j]].neighbor_markers[search_reverse[j]]=w<0 ? pm::WEIGHT_SIGN : 0;
                }
            }
            if (w<0) {
                negative[u[j]]^=1;
                if (v[j]>=0) negative[v[j]]^=1;
                flooder.negative_weight_obs_mask^=logical[j];
                flooder.negative_weight_sum+=w;
            }
        }
        if (out_edges) for (int j:edge_order)
            if (matcher->search_flooder.graph.nodes[u[j]].neighbor_markers[search_forward[j]]&pm::WEIGHT_SIGN)
                matcher->search_flooder.graph.negative_weight_edges.push_back({u[j],v[j]<0 ? SIZE_MAX : (size_t)v[j]});
        events.clear();
        for (int i=0;i<n;++i) {
            if (negative[i]) flooder.negative_weight_detection_events.push_back(i);
            if (syndrome[i]) events.push_back(i);
        }
        // Match upstream's event insertion order for negative-weight handling.
        for (auto i:events) {
            if (!negative[i]) solver.create_detection_event(&flooder.graph.nodes[i]);
            else negative[i]=0;
        }
        for (auto i:flooder.negative_weight_detection_events)
            if (negative[i]) solver.create_detection_event(&flooder.graph.nodes[i]);
        while (true) {
            auto event=flooder.run_until_next_mwpm_notification();
            if (event.event_type==pm::NO_EVENT) break;
            solver.process_event(event);
        }
        if (solver.node_arena.allocated.size()!=solver.node_arena.available.size()) {
            solver.reset();
            throw std::runtime_error("syndrome has no perfect matching");
        }
        if (out_edges) {extract_edges(*out_edges);return;}
        pm::MatchingResult result;
        auto extract=[&](const std::vector<uint64_t>& detectors) {
            for (auto i:detectors) if (flooder.graph.nodes[i].region_that_arrived)
                result+=solver.shatter_blossom_and_extract_matches(flooder.graph.nodes[i].region_that_arrived_top);
        };
        extract(events);
        extract(flooder.negative_weight_detection_events);
        result.obs_mask^=flooder.negative_weight_obs_mask;
        result.weight+=flooder.negative_weight_sum;
        *label=(uint8_t)(result.obs_mask&1);
        *weight=(double)result.weight/(2*scale);
        double original=0;
        for (int id:baseline) original+=weights[id];
        *gain=original-*weight;
    }
};

API void* latency_create(int n,int m,const int32_t* u,const int32_t* v,const uint8_t* logical,
    int others,int maps,const uint64_t* keys,const int32_t* ids) {
    try { return new FixedMatcher(n,m,u,v,logical,others,maps,keys,ids); }
    catch (const std::exception& e) {last_error=e.what();return nullptr;}
}
API void latency_destroy(void* handle) {delete (FixedMatcher*)handle;}
API const char* latency_error() {return last_error.c_str();}
API int latency_prepare(void* handle,const int64_t* pairs,int count,uint8_t* parity) {
    try {((FixedMatcher*)handle)->prepare(pairs,count,parity);return 0;}
    catch (const std::exception& e) {last_error=e.what();return -1;}
}
API int latency_solve(void* handle,const double* weights,const uint8_t* syndrome,
    uint8_t* label,double* weight,double* gain) {
    try {((FixedMatcher*)handle)->solve(weights,syndrome,label,weight,gain);return 0;}
    catch (const std::exception& e) {last_error=e.what();return -1;}
}
API int latency_solve_edges(void* handle,const double* weights,const uint8_t* syndrome,
    int64_t* out,int capacity,int* count) {
    try {
        std::vector<int64_t> edges;
        uint8_t label=0;double weight=0,gain=0;
        ((FixedMatcher*)handle)->solve(weights,syndrome,&label,&weight,&gain,&edges);
        if (edges.size()/2>(size_t)capacity) throw std::runtime_error("edge output buffer too small");
        std::copy(edges.begin(),edges.end(),out);*count=edges.size()/2;return 0;
    } catch (const std::exception& e) {last_error=e.what();return -1;}
}
API int latency_events(void* handle,const int64_t* pairs,int count,uint8_t* out) {
    try {((FixedMatcher*)handle)->joint_events(pairs,count,out);return 0;}
    catch (const std::exception& e) {last_error=e.what();return -1;}
}
