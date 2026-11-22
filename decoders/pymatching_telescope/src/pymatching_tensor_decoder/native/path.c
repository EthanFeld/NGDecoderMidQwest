/* Nonnegative path score through the memory matching graph. Used correction
 * edges cost zero; all other edges keep their original nonnegative cost.
 * This is a routing heuristic, not a logical likelihood or ML certificate. */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct { double cost; int node; } PathEntry;

static void push(PathEntry *heap, int *size, PathEntry entry) {
    int i=(*size)++;
    while (i) {
        int parent=(i-1)/2;
        if (heap[parent].cost<=entry.cost) break;
        heap[i]=heap[parent]; i=parent;
    }
    heap[i]=entry;
}

static PathEntry pop(PathEntry *heap, int *size) {
    PathEntry result=heap[0], entry=heap[--(*size)];
    if (*size) {
        int i=0;
        while (2*i+1<*size) {
            int child=2*i+1;
            if (child+1<*size && heap[child+1].cost<heap[child].cost) ++child;
            if (entry.cost<=heap[child].cost) break;
            heap[i]=heap[child]; i=child;
        }
        heap[i]=entry;
    }
    return result;
}

API int path_proxy_abi_version(void) { return 1; }

API int path_proxy(int original_nodes, int nodes, int edges, int count,
                   const int32_t *mapping, const uint8_t *boundary_logical,
                   const int32_t *start, const int32_t *target, const int32_t *ids,
                   const double *weights, const int64_t *pairs,
                   double *score, uint8_t *label) {
    if (nodes<2 || edges<1 || count<0) return -2;
    double *distance=malloc((size_t)nodes*sizeof(double));
    uint8_t *used=calloc((size_t)edges,1);
    PathEntry *heap=malloc(((size_t)2*edges+1)*sizeof(PathEntry));
    if (!distance || !used || !heap) {
        free(distance); free(used); free(heap); return -3;
    }
    *label=0;
    for (int i=0; i<count; ++i) {
        int64_t oa=pairs[2*i], ob=pairs[2*i+1];
        if (oa<0 || oa>=original_nodes || ob>=original_nodes || ob< -1) {
            free(distance); free(used); free(heap); return -2;
        }
        if (ob<0) *label ^= boundary_logical[oa];
        int a=mapping[oa];
        if (a<0) continue;
        int b=ob<0 ? nodes-1-boundary_logical[oa] : mapping[ob];
        if (b<0) { free(distance); free(used); free(heap); return -2; }
        int found=0;
        for (int j=start[a]; j<start[a+1]; ++j) if (target[j]==b) {
            used[ids[j]]^=1; found=1; break;
        }
        if (!found) { free(distance); free(used); free(heap); return -2; }
    }
    for (int i=0; i<nodes; ++i) distance[i]=INFINITY;
    int source=nodes-2, destination=nodes-1, size=0;
    distance[source]=0;
    push(heap,&size,(PathEntry){0,source});
    while (size) {
        PathEntry entry=pop(heap,&size);
        if (entry.cost!=distance[entry.node]) continue;
        if (entry.node==destination) break;
        for (int j=start[entry.node]; j<start[entry.node+1]; ++j) {
            int v=target[j], id=ids[j];
            double next=entry.cost+(used[id] ? 0 : weights[id]);
            if (next<distance[v]) {
                distance[v]=next;
                push(heap,&size,(PathEntry){next,v});
            }
        }
    }
    *score=distance[destination];
    free(distance); free(used); free(heap); return 0;
}
