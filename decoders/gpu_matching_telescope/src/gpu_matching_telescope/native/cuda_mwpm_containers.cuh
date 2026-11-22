// Device-side allocation and containers for the pinned sparse-blossom core.
#pragma once

namespace gpu {

template <class T> struct remove_reference { using type = T; };
template <class T> struct remove_reference<T&> { using type = T; };
template <class T> struct remove_reference<T&&> { using type = T; };
template <class T> using remove_reference_t = typename remove_reference<T>::type;

template <class T>
__device__ constexpr remove_reference_t<T>&& move(T&& value) noexcept {
    return static_cast<remove_reference_t<T>&&>(value);
}

template <class T>
__device__ constexpr T&& forward(remove_reference_t<T>& value) noexcept {
    return static_cast<T&&>(value);
}

template <bool, class T = void> struct enable_if {};
template <class T> struct enable_if<true, T> { using type = T; };

template <class T> struct is_integral { static constexpr bool value = false; };
template <> struct is_integral<bool> { static constexpr bool value = true; };
template <> struct is_integral<char> { static constexpr bool value = true; };
template <> struct is_integral<signed char> { static constexpr bool value = true; };
template <> struct is_integral<unsigned char> { static constexpr bool value = true; };
template <> struct is_integral<short> { static constexpr bool value = true; };
template <> struct is_integral<unsigned short> { static constexpr bool value = true; };
template <> struct is_integral<int> { static constexpr bool value = true; };
template <> struct is_integral<unsigned int> { static constexpr bool value = true; };
template <> struct is_integral<long> { static constexpr bool value = true; };
template <> struct is_integral<unsigned long> { static constexpr bool value = true; };
template <> struct is_integral<long long> { static constexpr bool value = true; };
template <> struct is_integral<unsigned long long> { static constexpr bool value = true; };

template <class T> struct is_unsigned { static constexpr bool value = T(-1) > T(0); };

template <class T>
struct numeric_limits {
    __device__ static constexpr T max() noexcept {
        return T(~0ull >> (64 - sizeof(T) * 8 + !is_unsigned<T>::value));
    }
};

template <class T>
__device__ constexpr const T& min(const T& a, const T& b) noexcept {
    return b < a ? b : a;
}

template <class T>
__device__ constexpr const T& max(const T& a, const T& b) noexcept {
    return a < b ? b : a;
}

template <class T>
__device__ constexpr T abs(T value) noexcept {
    return value < 0 ? -value : value;
}

template <class T>
__device__ void swap(T& a, T& b) {
    T previous = gpu::move(a);
    a = gpu::move(b);
    b = gpu::move(previous);
}

template <class T>
__device__ constexpr auto distance(T* first, T* last) noexcept {
    return first == last ? 0 : last - first;
}

template <class T, class Predicate>
__device__ T* find_if(T* first, T* last, Predicate predicate) {
    while (first != last && !predicate(*first)) {
        ++first;
    }
    return first;
}

__device__ inline int bit_width(unsigned long long value) noexcept {
    return 64 - __clzll(value);
}

template <class First, class Second>
struct pair {
    First first;
    Second second;
};

template <class T, size_t N>
struct array {
    T items[N];

    __device__ constexpr size_t size() const noexcept { return N; }
    __device__ T& operator[](size_t index) noexcept { return items[index]; }
    __device__ const T& operator[](size_t index) const noexcept { return items[index]; }
};

[[noreturn]] __device__ inline void fail() {
    // A completion count written by the caller distinguishes finished lanes
    // from exhausted workspaces. The host owns and can reuse the workspace.
    asm volatile("exit;");
    __builtin_unreachable();
}

inline constexpr unsigned workspace_slots = 256;

struct Workspace {
    unsigned char* base;
    unsigned int bytes;
    unsigned int used;
    unsigned int free_heads[31];
};

__device__ Workspace workspaces[workspace_slots];

__device__ inline Workspace& workspace() {
    const unsigned lane = blockDim.x * blockIdx.x + threadIdx.x;
    if (lane >= workspace_slots) {
        fail();
    }
    return workspaces[lane];
}

__device__ inline void set_workspace(void* base, size_t bytes) {
    auto& pool = workspace();
    if (!base || bytes < 64 || bytes > 0xffffffffu) {
        fail();
    }
    pool.base = static_cast<unsigned char*>(base);
    pool.bytes = static_cast<unsigned>(bytes);
    pool.used = 16;
    for (auto& offset : pool.free_heads) {
        offset = 0;
    }
}

__device__ inline void* allocate(size_t bytes) {
    auto& pool = workspace();
    // Each allocation has an aligned 16-byte header. Preserve it when the
    // caller placement-constructs an object in the returned storage.
    if (!pool.base || bytes > (1u << 30) - 16) {
        fail();
    }
    const unsigned required = max(static_cast<unsigned>(bytes) + 16, 32u);
    const unsigned order = 32 - __clz(required - 1);
    unsigned offset = pool.free_heads[order];
    if (offset) {
        pool.free_heads[order] = *reinterpret_cast<unsigned*>(pool.base + offset);
    } else {
        const unsigned capacity = 1u << order;
        if (capacity > pool.bytes - pool.used) {
            fail();
        }
        offset = pool.used;
        pool.used += capacity;
    }
    auto* header = reinterpret_cast<unsigned*>(pool.base + offset);
    header[1] = order;
    return pool.base + offset + 16;
}

__device__ inline void deallocate(void* memory) {
    if (!memory) {
        return;
    }
    auto& pool = workspace();
    auto* block = static_cast<unsigned char*>(memory) - 16;
    auto* header = reinterpret_cast<unsigned*>(block);
    auto& head = pool.free_heads[header[1]];
    header[0] = head;
    head = static_cast<unsigned>(block - pool.base);
}

template <class T>
class vector {
    // The sentinel distinguishes graph arrays borrowed from per-shot workspace
    // from the dynamic, owning vectors used by the matcher and its radix heap.
    static constexpr unsigned borrowed = 0xffffffffu;
    T* storage = nullptr;
    unsigned count = 0;
    unsigned capacity = 0;

    __device__ void release() {
        if (capacity != borrowed) {
            clear();
            deallocate(storage);
        }
        storage = nullptr;
        count = 0;
        capacity = 0;
    }

    __device__ void transfer(vector& other) noexcept {
        storage = other.storage;
        count = other.count;
        capacity = other.capacity;
        other.storage = nullptr;
        other.count = 0;
        other.capacity = 0;
    }

    __device__ void grow() {
        if (capacity == borrowed) {
            fail();
        }
        if (count == capacity) {
            reserve(capacity ? 2 * static_cast<size_t>(capacity) : 4);
        }
    }

public:
    __device__ vector() {}
    __device__ explicit vector(size_t count) { resize(count); }

    __device__ vector(vector&& other) noexcept { transfer(other); }
    __device__ vector& operator=(vector&& other) noexcept {
        if (this != &other) {
            release();
            transfer(other);
        }
        return *this;
    }

    __device__ vector(const vector& other) {
        reserve(other.count);
        for (const auto& value : other) {
            new (storage + count++) T(value);
        }
    }

    __device__ vector& operator=(const vector& other) {
        if (this != &other) {
            vector copy(other);
            *this = gpu::move(copy);
        }
        return *this;
    }

    __device__ ~vector() { release(); }

    __device__ void bind(T* values, size_t size) {
        if (size >= borrowed || (size && !values)) {
            fail();
        }
        release();
        storage = values;
        count = static_cast<unsigned>(size);
        capacity = borrowed;
    }

    __device__ size_t size() const noexcept { return count; }
    __device__ bool empty() const noexcept { return count == 0; }
    __device__ T* data() noexcept { return storage; }
    __device__ const T* data() const noexcept { return storage; }
    __device__ T* begin() noexcept { return storage; }
    __device__ const T* begin() const noexcept { return storage; }
    __device__ T* end() noexcept { return count ? storage + count : storage; }
    __device__ const T* end() const noexcept { return count ? storage + count : storage; }
    __device__ T& operator[](size_t index) noexcept { return storage[index]; }
    __device__ const T& operator[](size_t index) const noexcept { return storage[index]; }
    __device__ T& back() noexcept { return storage[count - 1]; }
    __device__ const T& back() const noexcept { return storage[count - 1]; }

    __device__ bool operator==(const vector& other) const {
        if (count != other.count) {
            return false;
        }
        for (unsigned i = 0; i < count; ++i) {
            if (!(storage[i] == other.storage[i])) {
                return false;
            }
        }
        return true;
    }

    __device__ bool operator!=(const vector& other) const { return !(*this == other); }

    __device__ void reserve(size_t size) {
        if (size <= count || (capacity != borrowed && size <= capacity)) {
            return;
        }
        if (capacity == borrowed || size > ((1u << 30) - 16) / sizeof(T)) {
            fail();
        }
        auto* grown = static_cast<T*>(allocate(size * sizeof(T)));
        for (unsigned i = 0; i < count; ++i) {
            new (grown + i) T(gpu::move(storage[i]));
            storage[i].~T();
        }
        deallocate(storage);
        storage = grown;
        capacity = static_cast<unsigned>(size);
    }

    __device__ void clear() {
        if (capacity == borrowed) {
            fail();
        }
        while (count) {
            storage[--count].~T();
        }
    }

    __device__ void resize(size_t size) {
        if (capacity == borrowed && size != count) {
            fail();
        }
        reserve(size);
        while (count > size) {
            storage[--count].~T();
        }
        while (count < size) {
            new (storage + count++) T();
        }
    }

    __device__ void pop_back() {
        if (capacity == borrowed || !count) {
            fail();
        }
        storage[--count].~T();
    }

    __device__ void push_back(const T& value) {
        if (count == capacity) {
            T copy = value;
            grow();
            new (storage + count++) T(gpu::move(copy));
        } else {
            grow();
            new (storage + count++) T(value);
        }
    }

    __device__ void push_back(T&& value) {
        if (count == capacity) {
            T copy = gpu::move(value);
            grow();
            new (storage + count++) T(gpu::move(copy));
        } else {
            grow();
            new (storage + count++) T(gpu::move(value));
        }
    }

    template <class... Args>
    __device__ void emplace_back(Args&&... args) {
        if (count == capacity) {
            T value(gpu::forward<Args>(args)...);
            grow();
            new (storage + count++) T(gpu::move(value));
        } else {
            grow();
            new (storage + count++) T(gpu::forward<Args>(args)...);
        }
    }
};

template <class T>
__device__ auto begin(T& values) -> decltype(values.begin()) { return values.begin(); }

template <class T>
__device__ auto end(T& values) -> decltype(values.end()) { return values.end(); }

template <class Container>
struct back_insert_iterator {
    Container* target;

    __device__ back_insert_iterator& operator*() noexcept { return *this; }
    __device__ back_insert_iterator& operator++() noexcept { return *this; }

    template <class T>
    __device__ back_insert_iterator& operator=(T&& value) {
        target->push_back(gpu::forward<T>(value));
        return *this;
    }
};

template <class Container>
__device__ back_insert_iterator<Container> back_inserter(Container& target) {
    return {&target};
}

template <class Input, class Output>
__device__ Output move(Input first, Input last, Output out) {
    while (first != last) {
        *out = gpu::move(*first);
        ++first;
        ++out;
    }
    return out;
}

template <class T>
__device__ void sift_down(T* values, size_t root, size_t count) {
    while (root < count / 2) {
        size_t child = root * 2 + 1;
        if (child + 1 < count && values[child] < values[child + 1]) {
            ++child;
        }
        if (!(values[root] < values[child])) {
            break;
        }
        gpu::swap(values[root], values[child]);
        root = child;
    }
}

template <class T>
__device__ void sort(T* first, T* last) {
    size_t count = gpu::distance(first, last);
    for (size_t i = count / 2; i; --i) {
        sift_down(first, i - 1, count);
    }
    while (count > 1) {
        gpu::swap(first[0], first[--count]);
        sift_down(first, 0, count);
    }
}

}  // namespace gpu
