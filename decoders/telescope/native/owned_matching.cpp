// Original telescope T-join solver. Standard-library-only; no decoder-library code.
#include "owned_mwpm.h"
#include "owned_queue.h"
#include "owned_workers.h"
#include "owned_certificate.h"
#include "owned_correlated.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr double INF = std::numeric_limits<double>::infinity();
thread_local std::string error;
using Item = std::pair<double, int>;
// Error-free TwoSum identifies the rounding direction. Preserve exact sums;
// widen only inexact ones so equality can certify a dual bound safely.
long double add_down(long double a,long double b) {
    long double sum=a+b, z=sum-a;
    long double residual=(a-(sum-z))+(b-z);
    return residual<0 ? std::nextafter(sum,-std::numeric_limits<long double>::infinity()) : sum;
}
long double add_up(long double a,long double b) {
    long double sum=a+b, z=sum-a;
    long double residual=(a-(sum-z))+(b-z);
    return residual>0 ? std::nextafter(sum,std::numeric_limits<long double>::infinity()) : sum;
}
struct Route { std::vector<double> distance; std::vector<int> edge; };
using MinQueue=owned_matching::MonotoneQueue;
struct PartialRoute {
    Route route;
    MinQueue queue;
    size_t bytes() const {return sizeof(PartialRoute)+route.distance.capacity()*sizeof(double)+route.edge.capacity()*sizeof(int)+queue.bytes();}
};
using Clock=std::chrono::steady_clock;
struct ScopeTime {
    std::atomic<int64_t>& output; Clock::time_point start=Clock::now();
    explicit ScopeTime(std::atomic<int64_t>& output):output(output){}
    ~ScopeTime(){output.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count(),std::memory_order_relaxed);}
};
struct Graph {
    int n, m;
    bool cache;
    std::vector<int> u, v, starts, adjacent, targets, parent, order, changed_reference;
    std::vector<double> adjacent_weights;
    std::vector<double> weights, absolute;
    std::vector<uint8_t> negative;
    std::vector<std::unique_ptr<Route>> routes;
    std::vector<std::unique_ptr<PartialRoute>> partials;
    std::vector<uint64_t> partial_used;
    std::atomic<uint64_t> partial_clock{0};
    size_t partial_budget=0;
    bool lazy_metric=false;
    std::vector<int> hint_order;
    Graph* reference=nullptr;
    bool reference_usable=false;
    std::unique_ptr<owned_matching::PathWorkers> path_pool;
    int path_worker_count=1;
    bool phase_pricing=false;
    int zero_edges=0;
    std::atomic<int64_t> certificate_rejections{0};
    std::atomic<int64_t> solves{0},certificates{0},blossoms{0},searches{0},cache_hits{0},seeded{0},seed_certificates{0};
    std::atomic<int64_t> initial_pairs{0},hinted_pairs{0},augmentations{0},sparse_rounds{0},priced_edges{0},dense_fallbacks{0};
    std::atomic<int64_t> incremental_searches{0};
    std::atomic<int64_t> phase_ns[12]{};
    Graph(int count, int edges, const int32_t* a, const int32_t* b, bool keep)
        : n(count), m(edges), cache(keep), u(a,a+edges), v(b,b+edges), starts(n+2,0),
          parent(n+1,-1), weights(m,INF), absolute(m), negative(m), routes(n+1), partials(n+1), partial_used(n+1) {
        if(n<0 || m<0) throw std::invalid_argument("negative graph size");
        for(int e=0;e<m;++e) {
            if(v[e]<0) v[e]=n;
            if(u[e]<0 || u[e]>=n || v[e]>n || u[e]==v[e])
                throw std::invalid_argument("invalid graph endpoint");
            ++starts[u[e]+1]; ++starts[v[e]+1];
        }
        for(int i=1;i<=n+1;++i) starts[i]+=starts[i-1];
        adjacent.resize(2*m); auto cursor=starts;
        for(int e=0;e<m;++e) { adjacent[cursor[u[e]]++]=e; adjacent[cursor[v[e]]++]=e; }
        targets.resize(2*m);adjacent_weights.resize(2*m);
        for(int a=0;a<=n;++a) for(int z=starts[a];z<starts[a+1];++z) targets[z]=neighbor(adjacent[z],a);
        // A deterministic forest provides a linear-time feasible completion for FPGA hints.
        std::vector<uint8_t> visited(n+1,0);
        for(int root=n;root>=0;--root) if(!visited[root]) {
            visited[root]=1; size_t begin=order.size(); order.push_back(root);
            for(size_t k=begin;k<order.size();++k) {
                int a=order[k];
                for(int z=starts[a];z<starts[a+1];++z) {
                    int e=adjacent[z], b=u[e]^v[e]^a;
                    if(!visited[b]) { visited[b]=1; parent[b]=e; order.push_back(b); }
                }
            }
        }
    }
    int neighbor(int e,int a) const { return u[e]^v[e]^a; }
    template<class Job> void sources(int count,Job&& job) {
        // Only independent updated-weight source rows use the shared pool.
        if(reference && path_pool && path_worker_count>1 && count>=16) path_pool->run(count,std::forward<Job>(job));
        else for(int i=0;i<count;++i) job(i);
    }
    void set_weights(const double* w) {
        bool changed=false;
        for(int e=0;e<m;++e) {
            if(!std::isfinite(w[e])) throw std::invalid_argument("nonfinite edge weight");
            changed |= weights[e]!=w[e]; weights[e]=w[e];
            absolute[e]=std::abs(w[e]); negative[e]=w[e]<0;
        }
        if(changed) { for(auto& r:routes) r.reset();for(auto& r:partials) r.reset(); hint_order.clear(); }
        if(changed) for(int z=0;z<2*m;++z) adjacent_weights[z]=absolute[adjacent[z]];
        if(changed) zero_edges=(int)std::count(absolute.begin(),absolute.end(),0.0);
        reference_usable=reference!=nullptr;
        changed_reference.clear();
        if(reference) for(int e=0;e<m;++e)
            if(!std::isfinite(reference->weights[e]) || absolute[e]>reference->absolute[e]) {reference_usable=false;break;}
            else if(absolute[e]<reference->absolute[e]) changed_reference.push_back(e);
    }
    Route shortest(int source) {
        ScopeTime clock(phase_ns[7]);
        ++searches;
        Route r;
        MinQueue queue;
        if(reference_usable && reference->routes[source]) {
            // Decreasing edge costs can only shorten prior paths. Relax all
            // changed edges, then propagate improvements with Dijkstra.
            r=*reference->routes[source]; ++incremental_searches;
            for(int e:changed_reference) {
                int a=u[e],b=v[e];
                if(r.distance[a]+absolute[e]<r.distance[b]) {r.distance[b]=r.distance[a]+absolute[e];r.edge[b]=e;queue.emplace(r.distance[b],b);}
                if(r.distance[b]+absolute[e]<r.distance[a]) {r.distance[a]=r.distance[b]+absolute[e];r.edge[a]=e;queue.emplace(r.distance[a],a);}
            }
        } else {
            r={std::vector<double>(n+1,INF),std::vector<int>(n+1,-1)};
            r.distance[source]=0; queue.emplace(0,source);
        }
        while(!queue.empty()) {
            auto [d,a]=queue.top(); queue.pop(); if(d!=r.distance[a]) continue;
            for(int z=starts[a];z<starts[a+1];++z) {
                int e=adjacent[z], b=targets[z]; double next=d+adjacent_weights[z];
                if(next<r.distance[b]) { r.distance[b]=next; r.edge[b]=e; queue.emplace(next,b); }
            }
        }
        return r;
    }
    void discard_stale(PartialRoute& p) {
        while(!p.queue.empty() && p.queue.top().first!=p.route.distance[p.queue.top().second]) p.queue.pop();
    }
    void advance(PartialRoute& p,double limit,int maximum=-1) {
        ScopeTime clock(phase_ns[7]);
        int count=0;
        for(;;) {
            discard_stale(p);
            if(p.queue.empty() || p.queue.top().first>limit || (maximum>=0 && count>=maximum)) break;
            auto [d,a]=p.queue.top();p.queue.pop();++count;
            for(int z=starts[a];z<starts[a+1];++z) {
                int e=adjacent[z],b=targets[z];double next=d+adjacent_weights[z];
                if(next<p.route.distance[b]) {p.route.distance[b]=next;p.route.edge[b]=e;p.queue.emplace(next,b);}
            }
        }
        if(p.queue.empty()) p.queue.release_unused_capacity();
    }
    void full_source(int source) {
        if(routes[source]) return;
        if(partials[source]) {
            advance(*partials[source],INF);
            routes[source]=std::make_unique<Route>(partials[source]->route);
        } else routes[source]=std::make_unique<Route>(shortest(source));
    }
    PartialRoute& partial(int source,std::vector<std::unique_ptr<PartialRoute>>& temporary) {
        const bool keep=cache || partial_budget>0;
        auto& slot=keep ? partials[source] : temporary[source];
        if(keep) partial_used[source]=++partial_clock;
        if(slot) {++cache_hits;return *slot;}
        slot=std::make_unique<PartialRoute>();auto& p=*slot;
        if(cache && routes[source]) {p.route=*routes[source];++cache_hits;return p;}
        ++searches;
        if(reference_usable && reference->cache) reference->full_source(source);
        if(reference_usable && reference->routes[source]) {
            p.route=*reference->routes[source];++incremental_searches;
            for(int e:changed_reference) {
                int a=u[e],b=v[e];
                if(p.route.distance[a]+absolute[e]<p.route.distance[b]) {p.route.distance[b]=p.route.distance[a]+absolute[e];p.route.edge[b]=e;p.queue.emplace(p.route.distance[b],b);}
                if(p.route.distance[b]+absolute[e]<p.route.distance[a]) {p.route.distance[a]=p.route.distance[b]+absolute[e];p.route.edge[a]=e;p.queue.emplace(p.route.distance[a],a);}
            }
        } else {
            p.route={std::vector<double>(n+1,INF),std::vector<int>(n+1,-1)};
            p.route.distance[source]=0;p.queue.emplace(0,source);
        }
        return p;
    }
    void trim_partials() {
        if(cache) return;
        size_t bytes=0;std::vector<std::pair<uint64_t,int>> lru;
        for(int a=0;a<=n;++a) if(partials[a]) {bytes+=partials[a]->bytes();lru.emplace_back(partial_used[a],a);}
        if(bytes<=partial_budget) return;
        for(auto [used,a]:lru) {
            size_t old=partials[a]->bytes();partials[a]->queue.release_unused_capacity();
            bytes-=old-partials[a]->bytes();
        }
        if(bytes<=partial_budget) return;
        std::sort(lru.begin(),lru.end());
        for(auto [used,a]:lru) {if(bytes<=partial_budget) break;bytes-=partials[a]->bytes();partials[a].reset();}
    }
    static double outward(double value) {
        return std::nextafter(value+1e-11*(1+std::abs(value)),INF);
    }
    void reach(PartialRoute& p,int target) {
        for(;;) {
            discard_stale(p);
            if(p.queue.empty() || p.queue.top().first>p.route.distance[target]) break;
            advance(p,p.route.distance[target],32);
        }
    }
    double solve_lazy(const uint8_t* syndrome,const uint8_t* seed,uint8_t* selected,const std::vector<int>& defects) {
        const int k=(int)defects.size();
        std::vector<std::vector<double>> metric(k,std::vector<double>(k,INF));
        std::vector<std::unique_ptr<PartialRoute>> temporary(n+1);
        std::vector<PartialRoute*> paths(k);
        auto start=Clock::now();
        auto mark=[&](int phase) {auto end=Clock::now();phase_ns[phase]+=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();start=end;};
        sources(k,[&](int i) {paths[i]=&partial(defects[i],temporary);metric[i][i]=0;});
        // All symmetric costs use the same higher-index source as the original
        // full metric. This preserves floating addition and distance tie order.
        auto pair=[&](int i,int j) {
            int a=std::max(i,j),b=std::min(i,j);
            if(a==b || std::isfinite(metric[a][b])) return;
            reach(*paths[a],defects[b]);
            metric[a][b]=metric[b][a]=paths[a]->route.distance[defects[b]];
        };
        // Resolve each exact nearest-neighbor row, including the complete tie
        // shell. Dijkstra state is retained and resumed for later dual queries.
        auto resolve_nearest=[&](int requested) {
        const int nearest=std::min(requested,k-1);
        std::vector<std::vector<int>> candidates(k), targets_by_source(k);
        sources(k,[&](int i) {
            auto& p=*paths[i];double limit=INF;
            for(;;) {
                discard_stale(p);
                double frontier=p.queue.empty() ? INF : p.queue.top().first;
                std::vector<double> known;known.reserve(k);
                for(int j=0;j<k;++j) if(j!=i && std::isfinite(p.route.distance[defects[j]])
                    && (p.queue.empty() || p.route.distance[defects[j]]<frontier)) known.push_back(p.route.distance[defects[j]]);
                if((int)known.size()>=nearest) {
                    std::nth_element(known.begin(),known.begin()+nearest-1,known.end());limit=outward(known[nearest-1]);advance(p,limit);break;
                }
                if(p.queue.empty()) break;
                advance(p,INF,32);
            }
            for(int j=0;j<k;++j) if(j!=i && std::isfinite(p.route.distance[defects[j]])
                && p.route.distance[defects[j]]<=limit) candidates[i].push_back(j);
        });
        for(int i=0;i<k;++i) for(int j:candidates[i]) targets_by_source[std::max(i,j)].push_back(std::min(i,j));
        sources(k,[&](int a) {
            auto& todo=targets_by_source[a];std::sort(todo.begin(),todo.end());todo.erase(std::unique(todo.begin(),todo.end()),todo.end());
            for(int b:todo) pair(a,b);
        });
        };
        resolve_nearest(12);
        mark(2);
        std::vector<int> preferred(k,-1);
        if(seed && valid(syndrome,seed)) {
            std::vector<int> component(n+1);for(int a=0;a<=n;++a) component[a]=a;
            auto root=[&](int a) {while(component[a]!=a) {component[a]=component[component[a]];a=component[a];}return a;};
            for(int e=0;e<m;++e) if(bool(seed[e])!=bool(negative[e])) component[root(u[e])]=root(v[e]);
            for(int i=0;i<k;++i) if(preferred[i]<0) {
                auto& p=*paths[i];double closest=INF;bool any=false;
                for(;;) {
                    closest=INF;any=false;
                    for(int j=i+1;j<k;++j) if(preferred[j]<0 && root(defects[i])==root(defects[j])) {
                        any=true;closest=std::min(closest,p.route.distance[defects[j]]);
                    }
                    discard_stale(p);
                    if(!any || p.queue.empty() || (std::isfinite(closest) && p.queue.top().first>outward(closest))) break;
                    advance(p,INF,32);
                }
                int best=-1;
                for(int j=i+1;j<k;++j) if(preferred[j]<0 && root(defects[i])==root(defects[j])
                    && std::isfinite(p.route.distance[defects[j]]) && p.route.distance[defects[j]]<=outward(closest)) {
                    pair(i,j);if(best<0 || metric[i][j]<metric[i][best]) best=j;
                }
                if(best>=0) {preferred[i]=best;preferred[best]=i;}
            }
        }
        mark(3);
        auto ensure=[&](const std::vector<double>& radii) {
            if(radii.empty()) {
                sources(k,[&](int i) {advance(*paths[i],INF);});
                for(int i=0;i<k;++i) for(int j=0;j<i;++j) pair(i,j);
            } else {
                double maximum=*std::max_element(radii.begin(),radii.end());
                sources(k,[&](int i) {advance(*paths[i],outward(radii[i]+maximum));});
                for(int i=0;i<k;++i) for(int j=0;j<i;++j) if(!std::isfinite(metric[i][j])
                    && paths[i]->route.distance[defects[j]]<=outward(radii[i]+radii[j])) pair(i,j);
            }
        };
        owned_matching::LazyMatchingOptions options;
        if(phase_pricing) {
            options.grow_nearest=resolve_nearest;
            options.price_geometry=[&](const std::vector<owned_matching::PricingPhase>& phases) {
                for(const auto& phase:phases) {
                    const int count=(int)phase.vertices.size();std::vector<double> limits(count,-1);
                    if(phase.colors.size()!=phase.vertices.size() || phase.offsets.size()!=phase.vertices.size()) return false;
                    for(auto offset:phase.offsets) if(!std::isfinite(offset)) return false;
                    for(int i=0;i<count;++i) {
                        int a=phase.vertices[i];long double maximum=-1;
                        for(int j=0;j<count;++j) {
                            int b=phase.vertices[j];
                            if(a>b && phase.colors[i]!=phase.colors[j] && !std::isfinite(metric[a][b]))
                                maximum=std::max(maximum,-(phase.offsets[i]+phase.offsets[j]));
                        }
                        if(maximum>=0) {double rounded=(double)maximum;if((long double)rounded<maximum) rounded=std::nextafter(rounded,INF);limits[i]=outward(rounded);}
                    }
                    sources(count,[&](int i) {if(limits[i]>=0) advance(*paths[phase.vertices[i]],limits[i]);});
                    for(int i=0;i<count;++i) if(limits[i]>=0) for(int j=0;j<count;++j) {
                        int a=phase.vertices[i],b=phase.vertices[j];
                        if(a<=b || phase.colors[i]==phase.colors[j] || std::isfinite(metric[a][b])) continue;
                        long double exact=-(phase.offsets[i]+phase.offsets[j]);
                        if(exact<0) continue;
                        double cutoff=(double)exact;if((long double)cutoff<exact) cutoff=std::nextafter(cutoff,INF);
                        if(paths[a]->route.distance[defects[b]]<=outward(cutoff)) pair(a,b);
                    }
                }
                return true;
            };
        }
        owned_matching::MatchingStats stats;
        auto mates=owned_matching::minimum_weight_perfect_matching_lazy(metric,ensure,&preferred,&stats,12,phase_pricing ? &options : nullptr);
        mark(4);
        phase_ns[8]+=stats.build_ns;phase_ns[9]+=stats.solve_ns;phase_ns[10]+=stats.pricing_ns;phase_ns[11]+=stats.prepare_ns;
        initial_pairs+=stats.initial_pairs;hinted_pairs+=stats.hinted_pairs;augmentations+=stats.augmentations;
        sparse_rounds+=stats.sparse_rounds;priced_edges+=stats.priced_edges;dense_fallbacks+=stats.dense_fallbacks;
        for(int i=0;i<k;++i) if(mates[i]>i) {
            reach(*paths[i],defects[mates[i]]);
            trace(paths[i]->route,defects[i],defects[mates[i]],selected);
        }
        if(!valid(syndrome,selected)) throw std::runtime_error("lazy matching returned inconsistent correction");
        trim_partials();mark(5);return cost(selected);
    }
    void trace(const Route& r,int source,int target,uint8_t* chosen) const {
        for(int steps=0;target!=source;++steps) {
            if(steps>n || r.edge[target]<0) throw std::runtime_error("unreachable correction path");
            int e=r.edge[target]; chosen[e]^=1; target=neighbor(e,target);
        }
    }
    bool valid(const uint8_t* syndrome,const uint8_t* selected) const {
        std::vector<uint8_t> parity(n,0);
        for(int e=0;e<m;++e) if(selected[e]) {
            parity[u[e]]^=1; if(v[e]<n) parity[v[e]]^=1;
        }
        return std::equal(parity.begin(),parity.end(),syndrome);
    }
    double cost(const uint8_t* selected) const {
        double result=0; for(int e=0;e<m;++e) if(selected[e]) result+=weights[e]; return result;
    }
    // Nearest-neighbor radii / 2 are a feasible dual of the metric matching LP.
    // A matching attaining their sum is globally optimal, including odd cycles.
    bool certificate(const std::vector<int>& defects,const uint8_t* syndrome,
                     const uint8_t* seed,uint8_t* selected) {
        int k=(int)defects.size(); if(!k) return false;
        using Bound=long double;
        using BoundItem=std::pair<Bound,int>;
        std::vector<Bound> distance(n+1,INF), nearest(k,INF);
        std::vector<int> owner(n+1,-1), predecessor(n+1,-1), partner(k,-1), left(k,-1), right(k,-1), bridge(k,-1);
        std::priority_queue<BoundItem,std::vector<BoundItem>,std::greater<BoundItem>> queue;
        for(int i=0;i<k;++i) { int a=defects[i]; distance[a]=0; owner[a]=i; queue.emplace(0,a); }
        while(!queue.empty()) {
            auto [d,a]=queue.top(); queue.pop(); if(d!=distance[a]) continue;
            for(int z=starts[a];z<starts[a+1];++z) {
                int e=adjacent[z], b=neighbor(e,a); Bound next=add_down(d,absolute[e]);
                if(next<distance[b]) {
                    distance[b]=next; owner[b]=owner[a]; predecessor[b]=e; queue.emplace(next,b);
                }
            }
        }
        for(int e=0;e<m;++e) {
            int a=u[e],b=v[e],i=owner[a],j=owner[b];
            if(i<0 || j<0 || i==j) continue;
            Bound d=add_down(add_down(distance[a],absolute[e]),distance[b]);
            if(d<nearest[i]) { nearest[i]=d;partner[i]=j;left[i]=a;right[i]=b;bridge[i]=e; }
            if(d<nearest[j]) { nearest[j]=d;partner[j]=i;left[j]=b;right[j]=a;bridge[j]=e; }
        }
        Bound bound=0;
        for(Bound d:nearest) {
            if(!std::isfinite(d)) return false;
            bound=add_down(bound,d*.5L);
        }
        auto certified=[&](const uint8_t* candidate) {
            if(!valid(syndrome,candidate)) return false;
            Bound upper=0;
            for(int e=0;e<m;++e) {
                if(candidate[e]>1) return false;
                if(bool(candidate[e])!=bool(negative[e])) upper=add_up(upper,absolute[e]);
            }
            // The dual uses lower bounds and the primal uses upper bounds.
            // A merely close cost is never accepted; uncertain sums use blossom.
            return upper<=bound;
        };
        // An FPGA-derived feasible seed can close the exact dual gap immediately.
        if(seed && certified(seed)) {
            std::memcpy(selected,seed,m); ++seed_certificates; return true;
        }
        std::vector<uint8_t> used(k,0), candidate(negative);
        for(int i=0;i<k;++i) if(!used[i]) {
            int j=partner[i];
            if(j<0 || used[j] || nearest[i]!=nearest[j]) return false;
            used[i]=used[j]=1; candidate[bridge[i]]^=1;
            for(int a=left[i]==defects[i] ? -1:left[i];a>=0;) {
                int e=predecessor[a]; if(e<0) break; candidate[e]^=1; a=neighbor(e,a);
            }
            for(int a=right[i]==defects[j] ? -1:right[i];a>=0;) {
                int e=predecessor[a]; if(e<0) break; candidate[e]^=1; a=neighbor(e,a);
            }
        }
        if(certified(candidate.data())) {
            std::memcpy(selected,candidate.data(),m); return true;
        }
        return false;
    }
    double solve(const uint8_t* syndrome,const double* w,const uint8_t* seed,uint8_t* selected) {
        auto start=Clock::now();
        auto mark=[&](int phase) {auto end=Clock::now();phase_ns[phase]+=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();start=end;};
        ++solves; if(seed) ++seeded; set_weights(w);
        std::vector<uint8_t> parity(n+1,0);
        for(int a=0;a<n;++a) { if(syndrome[a]>1) throw std::invalid_argument("nonbinary syndrome"); parity[a]=syndrome[a]; }
        for(int e=0;e<m;++e) if(negative[e]) { parity[u[e]]^=1; parity[v[e]]^=1; }
        parity[n]=0; for(int a=0;a<n;++a) parity[n]^=parity[a];
        std::vector<int> defects; for(int a=0;a<=n;++a) if(parity[a]) defects.push_back(a);
        std::copy(negative.begin(),negative.end(),selected);
        mark(0);
        if(defects.empty()) return cost(selected);
        bool obstructed=zero_edges && owned_matching::zero_component_obstructs_certificate(n,starts,targets,adjacent_weights,defects);
        if(obstructed) ++certificate_rejections;
        bool certified=!obstructed && certificate(defects,syndrome,seed,selected);mark(1);
        if(certified) { ++certificates; return cost(selected); }
        ++blossoms;
        if(lazy_metric) return solve_lazy(syndrome,seed,selected,defects);
        const int k=(int)defects.size();
        std::vector<std::vector<double>> metric(k,std::vector<double>(k,INF));
        std::vector<std::unique_ptr<Route>> temporary(k);
        std::vector<Route*> paths(k);
        for(int i=0;i<k;++i) {
            int source=defects[i];
            if(cache) {
                if(!routes[source]) routes[source]=std::make_unique<Route>(shortest(source)); else ++cache_hits;
                paths[i]=routes[source].get();
            } else { temporary[i]=std::make_unique<Route>(shortest(source)); paths[i]=temporary[i].get(); }
            metric[i][i]=0;
            for(int j=0;j<i;++j) metric[i][j]=metric[j][i]=paths[i]->distance[defects[j]];
        }
        mark(2);
        std::vector<int> preferred(k,-1);
        if(seed && valid(syndrome,seed)) {
            // Translate the feasible FPGA/first-pass chain into preferred pairs.
            // The blossom core uses them only when its dual proves them tight.
            std::vector<int> component(n+1);
            for(int a=0;a<=n;++a) component[a]=a;
            auto root=[&](int a) { while(component[a]!=a) {component[a]=component[component[a]];a=component[a];} return a; };
            for(int e=0;e<m;++e) if(bool(seed[e])!=bool(negative[e])) {
                int a=root(u[e]), b=root(v[e]); component[a]=b;
            }
            for(int i=0;i<k;++i) if(preferred[i]<0) {
                int best=-1;
                for(int j=i+1;j<k;++j) if(preferred[j]<0 && root(defects[i])==root(defects[j])
                    && (best<0 || metric[i][j]<metric[i][best])) best=j;
                if(best>=0) {preferred[i]=best;preferred[best]=i;}
            }
        }
        mark(3);
        owned_matching::MatchingStats stats;
        auto mates=owned_matching::minimum_weight_perfect_matching_sparse(metric,&preferred,&stats);
        mark(4);
        phase_ns[8]+=stats.build_ns;phase_ns[9]+=stats.solve_ns;
        phase_ns[10]+=stats.pricing_ns;phase_ns[11]+=stats.prepare_ns;
        initial_pairs+=stats.initial_pairs; hinted_pairs+=stats.hinted_pairs; augmentations+=stats.augmentations;
        sparse_rounds+=stats.sparse_rounds;priced_edges+=stats.priced_edges;dense_fallbacks+=stats.dense_fallbacks;
        for(int i=0;i<k;++i) if(mates[i]>i) trace(*paths[i],defects[i],defects[mates[i]],selected);
        if(!valid(syndrome,selected)) throw std::runtime_error("matching returned inconsistent correction");
        mark(5);
        return cost(selected);
    }
    // Use the removed detector pairs first, then cheaply complete the residual.
    // This constructs a feasible hint only; exact solving always verifies it.
    void hint(const uint8_t* syndrome,const uint8_t* removed,const double* w,uint8_t* selected) {
        ScopeTime clock(phase_ns[6]);
        set_weights(w); std::fill(selected,selected+m,0);
        std::vector<uint8_t> remaining(removed,removed+n); remaining.push_back(0);
        // Native neighboring pairs are the FPGA's most common partial decisions.
        if(hint_order.empty()) {
            hint_order.resize(m); for(int e=0;e<m;++e) hint_order[e]=e;
            std::stable_sort(hint_order.begin(),hint_order.end(),[&](int a,int b){return absolute[a]<absolute[b];});
        }
        for(int e:hint_order) if(v[e]<n && remaining[u[e]] && remaining[v[e]]) {
            remaining[u[e]]=remaining[v[e]]=0; selected[e]^=1;
        }
        // Recover short bridge pairs too; the UART gives their endpoints only.
        for(int a=0;a<n;++a) if(remaining[a]) {
            int best=-1, first=-1, second=-1; double cheapest=INF;
            for(int z=starts[a];z<starts[a+1];++z) {
                int e=adjacent[z], b=neighbor(e,a); if(b==n) continue;
                for(int zz=starts[b];zz<starts[b+1];++zz) {
                    int f=adjacent[zz], c=neighbor(f,b);
                    if(c<n && c!=a && remaining[c] && absolute[e]+absolute[f]<cheapest) {
                        best=c;first=e;second=f;cheapest=absolute[e]+absolute[f];
                    }
                }
            }
            if(best>=0) {remaining[a]=remaining[best]=0;selected[first]^=1;selected[second]^=1;}
        }
        std::vector<uint8_t> pending(syndrome,syndrome+n); pending.push_back(0);
        for(int e=0;e<m;++e) if(selected[e]) { pending[u[e]]^=1;pending[v[e]]^=1; }
        pending[n]=0; for(int a=0;a<n;++a) pending[n]^=pending[a];
        for(int a=0;a<=n;++a) if(pending[a]) {
            std::unique_ptr<Route> temporary;
            Route* path;
            if(cache) {
                if(!routes[a]) routes[a]=std::make_unique<Route>(shortest(a)); else ++cache_hits;
                path=routes[a].get();
            } else {temporary=std::make_unique<Route>(shortest(a));path=temporary.get();}
            int best=-1;
            for(int b=a+1;b<=n;++b) if(pending[b] && (best<0 || path->distance[b]<path->distance[best])) best=b;
            if(best<0 || !std::isfinite(path->distance[best])) throw std::runtime_error("hint has odd disconnected component");
            trace(*path,a,best,selected);pending[a]=pending[best]=0;
        }
        if(!valid(syndrome,selected)) throw std::runtime_error("invalid reconstructed FPGA hint");
    }
};
using CorrelatedEngine=owned_matching::CorrelatedPipeline<Graph>;
}

API const char* owned_error() { return error.c_str(); }
API void* owned_create(int n,int m,const int32_t* u,const int32_t* v,int cache) {
    try { return new Graph(n,m,u,v,cache!=0); } catch(const std::exception& e) {error=e.what();return nullptr;}
}
API void owned_destroy(void* p) { delete static_cast<Graph*>(p); }
API int owned_reference(void* p,void* reference) {
    try {
        auto& g=*static_cast<Graph*>(p);auto* r=static_cast<Graph*>(reference);
        if(r && (r==&g || g.n!=r->n || g.u!=r->u || g.v!=r->v)) throw std::invalid_argument("reference graph topology differs");
        g.reference=r;g.reference_usable=false;return 0;
    } catch(const std::exception& e) {error=e.what();return -1;}
}
API int owned_solve(void* p,const uint8_t* syndrome,const double* weights,const uint8_t* seed,uint8_t* output,double* cost) {
    try {*cost=static_cast<Graph*>(p)->solve(syndrome,weights,seed,output);return 0;}
    catch(const std::exception& e) {error=e.what();return -1;}
}
API int owned_hint(void* p,const uint8_t* syndrome,const uint8_t* removed,const double* weights,uint8_t* output) {
    try {static_cast<Graph*>(p)->hint(syndrome,removed,weights,output);return 0;}
    catch(const std::exception& e) {error=e.what();return -1;}
}
API void owned_stats(void* p,int64_t* output) {
    auto& g=*static_cast<Graph*>(p);
    int64_t s[]={g.solves,g.certificates,g.blossoms,g.searches,g.cache_hits,g.seeded,g.seed_certificates,
        g.initial_pairs,g.hinted_pairs,g.augmentations,g.sparse_rounds,g.priced_edges,g.dense_fallbacks,g.incremental_searches};
    std::copy(s,s+14,output);
}
API void owned_profile(void* p,int64_t* output) {
    auto& g=*static_cast<Graph*>(p);std::copy(g.phase_ns,g.phase_ns+12,output);
}
API int owned_profile_count() {return 12;}
API void owned_lazy_metric(void* p,int enabled) {static_cast<Graph*>(p)->lazy_metric=enabled!=0;}
API void owned_partial_cache_budget(void* p,int64_t bytes) {
    auto& g=*static_cast<Graph*>(p);g.partial_budget=bytes>0 ? (size_t)bytes:0;g.trim_partials();
}
API int owned_path_workers(void* p,int workers) {
    try {
        if(workers<1 || workers>32) throw std::invalid_argument("path workers must be between 1 and 32");
        auto& g=*static_cast<Graph*>(p);
        if(workers==1) {g.path_pool.reset();g.path_worker_count=1;}
        else {auto pool=std::make_unique<owned_matching::PathWorkers>(workers);g.path_worker_count=pool->workers();g.path_pool=std::move(pool);}
        return 0;
    } catch(const std::exception& e) {error=e.what();return -1;}
}
API int owned_path_worker_count(void* p) {return static_cast<Graph*>(p)->path_worker_count;}
API void owned_phase_pricing(void* p,int enabled) {static_cast<Graph*>(p)->phase_pricing=enabled!=0;}
API int64_t owned_certificate_rejections(void* p) {return static_cast<Graph*>(p)->certificate_rejections.load();}
API void* owned_correlated_create(void* p0,void* p1,void* u0,void* u1,int joint_n,
    const int32_t* indices0,const int32_t* indices1,const double* base,int rules,
    const int32_t* causes,const int32_t* targets,const double* weights,int workers) {
    try {
        return new CorrelatedEngine(static_cast<Graph*>(p0),static_cast<Graph*>(p1),
            static_cast<Graph*>(u0),static_cast<Graph*>(u1),joint_n,indices0,indices1,
            base,rules,causes,targets,weights,workers);
    } catch(const std::exception& e) {error=e.what();return nullptr;}
}
API void owned_correlated_destroy(void* p) {delete static_cast<CorrelatedEngine*>(p);}
API int owned_correlated_solve(void* p,const uint8_t* row,const uint8_t* removed,int correlations,
    uint8_t* output,int* hint_used) {
    try {static_cast<CorrelatedEngine*>(p)->solve(row,removed,correlations!=0,output,*hint_used);return 0;}
    catch(const std::exception& e) {error=e.what();return -1;}
}
API int owned_correlated_basis_workers(void* p,int workers) {
    try {static_cast<CorrelatedEngine*>(p)->set_basis_workers(workers);return 0;}
    catch(const std::exception& e) {error=e.what();return -1;}
}
API int owned_correlated_basis_worker_count(void* p) {return static_cast<CorrelatedEngine*>(p)->basis_workers();}
API void owned_correlated_snapshot(void* p,uint8_t* first,double* weights) {
    auto& engine=*static_cast<CorrelatedEngine*>(p);
    std::copy(engine.last_first().begin(),engine.last_first().end(),first);
    std::copy(engine.last_weights().begin(),engine.last_weights().end(),weights);
}
API void owned_thread_fp_probe(uint32_t* output) {
    auto main=owned_matching::FloatingEnvironment::capture();
    output[0]=main.x87_control;output[1]=main.sse_control;
    std::thread worker([&] {auto state=owned_matching::FloatingEnvironment::capture();output[2]=state.x87_control;output[3]=state.sse_control;});
    worker.join();
}
