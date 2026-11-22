/* Sparse positive tensor contraction with sector-protected, bounded truncation.
 * Scores guide retention; the accumulated masses always remain sum-product.
 * No hard decisions, labels, or sampled answers enter the contraction.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

#define LOGICAL (UINT64_C(1)<<63)
typedef __uint128_t Key;

static Key lane_bit(unsigned lane) {
    return (Key)1 << (lane + (lane>=63));
}

static Key words(const uint64_t *mask, int step) {
    return (Key)mask[2*step] | ((Key)mask[2*step+1]<<64);
}

typedef struct {
    int m, n, logical_offset, unused_count, lanes;
    const double *p;
    const uint64_t *toggle, *close;
    const int32_t *new_start, *new_detector;
    const uint8_t *new_lane;
    const double *future_odds;
    const uint8_t *detector_offset;
    const int32_t *unused;
    double zero_margin;
    int certificate_count;
    const uint64_t *certificate_keys;
    const uint8_t *certificate_labels;
    const double *certificate_margins;
} FrontierModel;

typedef struct { Key key; double mass, score; } Entry;
typedef struct {
    int cap, hash_size;
    uint32_t epoch;
    Key *hash_keys;
    double *hash_mass;
    uint32_t *stamp;
    int *touched;
    Entry *state, *heap;
} Workspace;

static uint64_t mix(uint64_t x) {
    x ^= x>>30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x>>27; x *= UINT64_C(0x94d049bb133111eb);
    return x^(x>>31);
}

static int better(Entry a, Entry b) {
    return a.score > b.score || (a.score == b.score && a.key < b.key);
}

static void keep(Entry *heap, int *count, int cap, Entry entry) {
    if (*count < cap) {
        int i=(*count)++;
        heap[i]=entry;
        while (i) {
            int parent=(i-1)/2;
            if (!better(heap[parent],heap[i])) break;
            Entry tmp=heap[parent]; heap[parent]=heap[i]; heap[i]=tmp;
            i=parent;
        }
    } else if (better(entry,heap[0])) {
        heap[0]=entry;
        int i=0;
        while (2*i+1 < *count) {
            int child=2*i+1;
            if (child+1 < *count && better(heap[child],heap[child+1])) ++child;
            if (!better(heap[i],heap[child])) break;
            Entry tmp=heap[i]; heap[i]=heap[child]; heap[child]=tmp;
            i=child;
        }
    }
}

static void destroy(Workspace *w) {
    free(w->hash_keys); free(w->hash_mass); free(w->stamp); free(w->touched);
    free(w->state); free(w->heap);
}

static int init(Workspace *w, int cap) {
    memset(w,0,sizeof(*w));
    w->cap=cap; w->hash_size=8;
    while (w->hash_size < 8*cap) w->hash_size *= 2;
    w->hash_keys=malloc((size_t)w->hash_size*sizeof(Key));
    w->hash_mass=malloc((size_t)w->hash_size*sizeof(double));
    w->stamp=calloc((size_t)w->hash_size,sizeof(uint32_t));
    w->touched=malloc((size_t)4*cap*sizeof(int));
    w->state=malloc((size_t)2*cap*sizeof(Entry));
    w->heap=malloc((size_t)2*cap*sizeof(Entry));
    if (!w->hash_keys || !w->hash_mass || !w->stamp || !w->touched || !w->state || !w->heap) {
        destroy(w); return -1;
    }
    return 0;
}

static void insert(Workspace *w, Key key, double mass, int *count) {
    int slot=(int)(mix((uint64_t)key ^ mix((uint64_t)(key>>64)))&
                   (uint64_t)(w->hash_size-1));
    while (w->stamp[slot]==w->epoch && w->hash_keys[slot]!=key)
        slot=(slot+1)&(w->hash_size-1);
    if (w->stamp[slot]!=w->epoch) {
        w->stamp[slot]=w->epoch; w->hash_keys[slot]=key; w->hash_mass[slot]=mass;
        w->touched[(*count)++]=slot;
    } else w->hash_mass[slot] += mass;
}

static int contract(const FrontierModel *m, const uint8_t *events, int beam,
                    Workspace *w, double *margin, int *pruned, int *numeric_loss) {
    int count=1;
    *pruned=0; *numeric_loss=0; *margin=0;
    w->state[0]=(Entry){0,1,1};
    for (int d=0; d<m->unused_count; ++d) {
        int detector=m->unused[d];
        if (((events[detector/8]^m->detector_offset[detector/8])>>(detector%8))&1) return -1;
    }
    for (int step=0; step<m->n; ++step) {
        Key initial=0;
        for (int j=m->new_start[step]; j<m->new_start[step+1]; ++j) {
            int detector=m->new_detector[j];
            if (((events[detector/8]^m->detector_offset[detector/8])>>(detector%8))&1)
                initial ^= lane_bit(m->new_lane[j]);
        }
        if (++w->epoch==0) {
            memset(w->stamp,0,(size_t)w->hash_size*sizeof(uint32_t)); w->epoch=1;
        }
        int touched=0;
        for (int i=0; i<count; ++i) {
            Key key=w->state[i].key^initial;
            double mass=w->state[i].mass;
            Key off=key, on=key^words(m->toggle,step);
            double off_mass=mass*(1-m->p[step]), on_mass=mass*m->p[step];
            if (mass>0 && ((on_mass==0 && m->p[step]>0) || off_mass==0)) *numeric_loss=1;
            Key close=words(m->close,step);
            if (!(off&close) && off_mass>0) insert(w,off,off_mass,&touched);
            if (!(on&close) && on_mass>0) insert(w,on,on_mass,&touched);
        }
        if (!touched) return -1;
        int sector_count[2]={0,0};
        for (int i=0; i<touched; ++i) ++sector_count[(w->hash_keys[w->touched[i]]&LOGICAL)!=0];
        int kept[2]={0,0};
        for (int i=0; i<touched; ++i) {
            int slot=w->touched[i];
            Key key=w->hash_keys[slot];
            double mass=w->hash_mass[slot], score=mass;
            int logical=(key&LOGICAL)!=0;
            if (sector_count[logical]>beam) {
                uint64_t residual=(uint64_t)key&~LOGICAL;
                while (residual) {
                    unsigned bit=(unsigned)__builtin_ctzll(residual);
                    score *= m->future_odds[(size_t)step*m->lanes+bit];
                    residual &= residual-1;
                }
                residual=(uint64_t)(key>>64);
                while (residual) {
                    unsigned bit=(unsigned)__builtin_ctzll(residual)+63;
                    score *= m->future_odds[(size_t)step*m->lanes+bit];
                    residual &= residual-1;
                }
                keep(w->heap+logical*w->cap,&kept[logical],beam,(Entry){key,mass,score});
            } else w->heap[logical*w->cap+kept[logical]++]=(Entry){key,mass,score};
        }
        *pruned += touched-kept[0]-kept[1];
        count=0;
        double largest=0;
        for (int l=0; l<2; ++l) for (int i=0; i<kept[l]; ++i) {
            Entry e=w->heap[l*w->cap+i];
            w->state[count++]=e;
            if (e.mass>largest) largest=e.mass;
        }
        for (int i=0; i<count; ++i) w->state[i].mass /= largest;
    }
    double p0=0,p1=0;
    for (int i=0; i<count; ++i) {
        if (w->state[i].key==0) p0+=w->state[i].mass;
        else if (w->state[i].key==LOGICAL) p1+=w->state[i].mass;
        else return -2;
    }
    if (p0+p1==0) return -1;
    *margin=fabs(p1-p0)/(p1+p0);
    int label=p0==p1 ? m->logical_offset : p1>p0;
    return label^m->logical_offset;
}

API int frontier_abi_version(void) { return 5; }

static void save_row(int shot, uint8_t label, double margin, int removed, int wide,
                     int failure, int loss, int proof, uint8_t *output, double *margins,
                     int32_t *pruned, uint8_t *refined, uint8_t *failed,
                     uint8_t *numeric_loss, uint8_t *certified) {
    output[shot]=label;
    if (margins) margins[shot]=margin;
    if (pruned) pruned[shot]=removed;
    if (refined) refined[shot]=(uint8_t)wide;
    if (failed) failed[shot]=(uint8_t)failure;
    if (numeric_loss) numeric_loss[shot]=(uint8_t)loss;
    if (certified) certified[shot]=(uint8_t)proof;
}

API int frontier_decode(const FrontierModel *m, const uint8_t *events, int shots, int width,
                        int beam, int refine_beam, double refine_margin, uint8_t *output,
                        double *margins, int32_t *pruned, uint8_t *refined, uint8_t *failed,
                        uint8_t *numeric_loss, uint8_t *certified) {
    if (beam<1 || refine_beam<beam || refine_beam>1024 || m->n<0 || m->m<0 ||
        m->lanes<1 || m->lanes>127) return -2;
    Workspace work;
    memset(&work,0,sizeof(work));
    int work_ready=0;
    int cache_size=8;
    while (cache_size<2*shots && cache_size<(1<<24)) cache_size*=2;
    int *cache=malloc((size_t)cache_size*sizeof(int));
    if (!cache) { destroy(&work); return -3; }
    for (int i=0; i<cache_size; ++i) cache[i]=-1;
    for (int shot=0; shot<shots; ++shot) {
        const uint8_t *row=events+(size_t)shot*width;
        uint64_t hash=UINT64_C(1469598103934665603);
        for (int b=0; b<width; ++b) { hash^=row[b]; hash*=UINT64_C(1099511628211); }
        int slot=(int)(mix(hash)&(uint64_t)(cache_size-1));
        while (cache[slot]>=0 && memcmp(row,events+(size_t)cache[slot]*width,(size_t)width))
            slot=(slot+1)&(cache_size-1);
        if (cache[slot]>=0) {
            int prev=cache[slot];
            output[shot]=output[prev];
            if (margins) margins[shot]=margins[prev];
            if (pruned) pruned[shot]=pruned[prev];
            if (refined) refined[shot]=refined[prev];
            if (failed) failed[shot]=failed[prev];
            if (numeric_loss) numeric_loss[shot]=numeric_loss[prev];
            if (certified) certified[shot]=certified[prev];
            continue;
        }
        if (m->certificate_count) {
            uint64_t address=0;
            for (int b=0; b<width; ++b)
                address |= ((uint64_t)(row[b]^m->detector_offset[b]))<<(8*b);
            int lo=0,hi=m->certificate_count;
            while (lo<hi) {
                int middle=lo+(hi-lo)/2;
                if (m->certificate_keys[middle]<address) lo=middle+1;
                else hi=middle;
            }
            if (lo<m->certificate_count && m->certificate_keys[lo]==address) {
                save_row(shot,m->certificate_labels[lo]^(uint8_t)m->logical_offset,
                         m->certificate_margins[lo],0,0,0,0,1,
                         output,margins,pruned,refined,failed,numeric_loss,certified);
                cache[slot]=shot;continue;
            }
        }
        int is_zero=m->zero_margin>0;
        for (int b=0; b<width && is_zero; ++b)
            if (row[b]!=m->detector_offset[b]) is_zero=0;
        if (is_zero) {
            save_row(shot,(uint8_t)m->logical_offset,m->zero_margin,0,0,0,0,1,
                     output,margins,pruned,refined,failed,numeric_loss,certified);
            cache[slot]=shot; continue;
        }
        if (!work_ready) {
            if (init(&work,refine_beam)) { free(cache); return -3; }
            work_ready=1;
        }
        int npruned=0, loss=0;
        double margin=0;
        int label=contract(m,row,beam,&work,&margin,&npruned,&loss);
        int wide=0;
        if (refine_beam>beam && (label<0 || margin<refine_margin)) {
            label=contract(m,row,refine_beam,&work,&margin,&npruned,&loss);
            wide=1;
        }
        save_row(shot,(uint8_t)(label<0 ? 0 : label),margin,npruned,wide,label<0,loss,0,
                 output,margins,pruned,refined,failed,numeric_loss,certified);
        cache[slot]=shot;
    }
    free(cache); destroy(&work); return 0;
}
