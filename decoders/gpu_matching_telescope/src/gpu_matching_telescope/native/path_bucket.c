/* Threshold-only path routing with a bounded fixed-point queue. */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

#define SCALE 64.0

typedef struct {
    int node;
    int next;
} BucketEntry;

static int mark_corrections(int original_nodes, int nodes, int count,
                            const int32_t *mapping, const uint8_t *boundary_logical,
                            const int32_t *start, const int32_t *target, const int32_t *ids,
                            const int64_t *pairs, uint8_t *used, uint8_t *label) {
    *label = 0;
    for (int i = 0; i < count; ++i) {
        int64_t left = pairs[2 * i], right = pairs[2 * i + 1];
        if (left < 0 || left >= original_nodes || right < -1 || right >= original_nodes) {
            return -2;
        }
        if (right < 0) {
            *label ^= boundary_logical[left];
        }
        int a = mapping[left];
        if (a < 0) {
            continue;
        }
        int b = right < 0 ? nodes - 1 - boundary_logical[left] : mapping[right];
        int j = start[a];
        while (j < start[a + 1] && target[j] != b) {
            ++j;
        }
        if (j == start[a + 1]) {
            return -2;
        }
        used[ids[j]] ^= 1;
    }
    return 0;
}

/*
 * Positive weights are floored at SCALE; corrected edges have exact weight zero.
 * Status 1 leaves the decision to the original floating-point implementation.
 */
API int path_bucket_route(int original_nodes, int nodes, int edges, int count,
                          const int32_t *mapping, const uint8_t *boundary_logical,
                          const int32_t *start, const int32_t *target, const int32_t *ids,
                          const uint32_t *weights, uint32_t min_positive,
                          const int64_t *pairs, double threshold,
                          uint8_t *routed, uint8_t *label) {
    if (nodes < 2 || edges < 1 || count < 0) {
        return -2;
    }
    if (!(threshold > 0 && threshold <= 1024) || nodes > 4098 || min_positive == 0) {
        return 1;
    }

    /* A shortest path uses at most nodes - 1 edges. Allow for rounding in its
     * FP64 sum, and round the computed search cutoff outward as well. */
    const double gamma = 2.0 * nodes * DBL_EPSILON;
    const double limit = threshold * SCALE;
    const int cap = (int)ceil(nextafter(limit / (1 - gamma), INFINITY));
    const int source = nodes - 2, destination = nodes - 1;
    int *distance = malloc((size_t)nodes * sizeof(int));
    int *head = malloc((size_t)cap * sizeof(int));
    uint8_t *used = calloc((size_t)edges, 1);
    BucketEntry *queue = malloc(((size_t)2 * edges + 1) * sizeof(BucketEntry));
    int code = 1, shortest = cap, queued = 1;

    if (!distance || !head || !used || !queue) {
        goto done;
    }
    code = mark_corrections(original_nodes, nodes, count, mapping, boundary_logical,
                            start, target, ids, pairs, used, label);
    if (code) {
        goto done;
    }
    for (int i = 0; i < nodes; ++i) {
        distance[i] = cap;
    }
    for (int i = 0; i < cap; ++i) {
        head[i] = -1;
    }
    distance[source] = 0;
    head[0] = 0;
    queue[0] = (BucketEntry){source, -1};

    for (int cost = 0; cost < cap; ++cost) {
        while (head[cost] >= 0) {
            BucketEntry entry = queue[head[cost]];
            head[cost] = entry.next;
            int a = entry.node;
            if (distance[a] != cost) {
                continue;
            }
            if (a == destination) {
                shortest = cost;
                goto done;
            }
            for (int j = start[a]; j < start[a + 1]; ++j) {
                int b = target[j], edge = ids[j];
                uint32_t weight = used[edge] ? 0 : weights[edge];
                if (weight >= (uint32_t)(cap - cost)) {
                    continue;
                }
                int next = cost + (int)weight;
                if (next < distance[b]) {
                    distance[b] = next;
                    queue[queued] = (BucketEntry){b, head[next]};
                    head[next] = queued++;
                }
            }
        }
    }

done:
    free(distance);
    free(head);
    free(used);
    free(queue);
    if (code) {
        return code;
    }
    if (shortest == cap) {
        *routed = 0;
        return 0;
    }
    /* Every positive edge adds less than one unit of quantization error. */
    const uint32_t positive_edges = (uint32_t)shortest / min_positive;
    const double upper = (shortest + positive_edges) * (1 + gamma);
    if (nextafter(upper, INFINITY) < limit) {
        *routed = 1;
        return 0;
    }
    return 1;
}
