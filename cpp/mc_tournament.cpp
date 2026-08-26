// Monte Carlo knockout-bracket simulator, with an exact DP cross-check.
//
// Build:  make -C cpp          (or: g++ -O2 -std=c++17 -pthread ...)
// Run:    ./cpp/mc_tournament --sims 100000000 --threads 8 --check
//
// Three things here are deliberate and are the parts worth arguing about:
//
// 1. DETERMINISM ACROSS THREAD COUNTS.  Work is split into fixed-size chunks
//    numbered 0..C-1, and chunk c is seeded from splitmix64(seed, c).  A chunk
//    therefore produces identical draws no matter which thread runs it or how
//    many threads exist, so --threads 1 and --threads 64 give bit-identical
//    output.  The obvious alternative -- give thread t the seed base+t -- makes
//    results depend on the scheduler, which turns any Monte Carlo bug into a
//    heisenbug you cannot reproduce.
//
// 2. SEEDING THROUGH SPLITMIX64.  Handing a generator nearby seeds (base+0,
//    base+1, ...) can produce correlated streams, because a raw seed is just
//    the initial state and small state differences take many steps to diffuse.
//    splitmix64 is an avalanche mixer: adjacent inputs give unrelated 256-bit
//    states.  This is the standard xoshiro seeding procedure and it is not
//    optional.
//
// 3. NO FALSE SHARING.  Each thread accumulates into its own local array and
//    merges once under a mutex at the end.  A shared counter array would put
//    several threads' counters on the same 64-byte cache line, and the
//    resulting ping-pong can make the parallel version slower than serial --
//    the classic way "I added threads and it got worse" happens.
//
// RNG choice: xoshiro256++ rather than xorshift128+.  xorshift128+ has known
// weakness in its low-order bits, which matters precisely because a Monte
// Carlo comparison `u < p` is sensitive to the whole mantissa.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Team { std::string name; double elo; };

// A 16-team bracket in bracket order (1 plays 2, 3 plays 4, ...).
// Ratings are produced by wcq/model/elo.py from real match history; regenerate
// with `python3 scripts/export_bracket.py`.
std::vector<Team> default_bracket() {
    return {
        {"Spain", 2130.5},      {"Morocco", 1893.9},
        {"France", 2039.6},     {"Japan", 1897.8},
        {"Argentina", 2123.5},  {"Mexico", 1885.0},
        {"England", 2024.3},    {"Switzerland", 1868.6},
        {"Brazil", 2031.2},     {"Ecuador", 1858.5},
        {"Portugal", 1987.1},   {"Italy", 1871.7},
        {"Colombia", 2001.6},   {"Germany", 1944.9},
        {"Netherlands", 1964.8},{"Belgium", 1956.6},
    };
}

inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct Xoshiro256pp {
    uint64_t s[4];

    explicit Xoshiro256pp(uint64_t seed) {
        uint64_t x = seed;
        for (int i = 0; i < 4; ++i) s[i] = splitmix64(x);
    }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    inline uint64_t next() {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;    s[3] = rotl(s[3], 45);
        return result;
    }
    // 53-bit mantissa in [0,1): the standard conversion, taking the HIGH bits.
    inline double uniform() { return double(next() >> 11) * 0x1.0p-53; }
};

inline double elo_win_prob(double a, double b) {
    return 1.0 / (1.0 + std::pow(10.0, (b - a) / 400.0));
}

// Precomputed P[i][j] = P(i beats j), flattened. One pow() per pair up front
// instead of one per simulated tie: for 100M sims that removes ~400M calls to
// a transcendental function, and it is the single biggest win in the program.
std::vector<double> build_win_matrix(const std::vector<Team>& t) {
    const size_t n = t.size();
    std::vector<double> w(n * n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            w[i * n + j] = elo_win_prob(t[i].elo, t[j].elo);
    return w;
}

// ---- exact answer, O(n^2), for cross-checking the simulator ---------------
std::vector<double> exact_title_probs(const std::vector<Team>& t) {
    const size_t n = t.size();
    const std::vector<double> w = build_win_matrix(t);
    std::vector<double> reach(n, 1.0), next(n);
    for (size_t block = 1; block < n; block *= 2) {
        for (size_t i = 0; i < n; ++i) {
            const size_t base = (i / block) * block;
            const size_t opp = ((i / block) % 2 == 0) ? base + block : base - block;
            double acc = 0.0;
            for (size_t j = opp; j < opp + block; ++j) acc += reach[j] * w[i * n + j];
            next[i] = reach[i] * acc;
        }
        reach.swap(next);
    }
    return reach;
}

constexpr long CHUNK = 100'000;   // simulations per work unit

void run_chunk(long chunk_id, long sims, const std::vector<double>& w, size_t n,
               uint64_t base_seed, std::vector<long>& out) {
    uint64_t mix = base_seed ^ (uint64_t)chunk_id;
    Xoshiro256pp rng(splitmix64(mix));
    std::vector<int> alive(n);
    for (long s = 0; s < sims; ++s) {
        for (size_t i = 0; i < n; ++i) alive[i] = (int)i;
        size_t remaining = n;
        while (remaining > 1) {
            for (size_t i = 0; i < remaining / 2; ++i) {
                const int a = alive[2 * i], b = alive[2 * i + 1];
                alive[i] = (rng.uniform() < w[(size_t)a * n + (size_t)b]) ? a : b;
            }
            remaining /= 2;
        }
        ++out[(size_t)alive[0]];
    }
}

}  // namespace

int main(int argc, char** argv) {
    long sims = 10'000'000;
    int threads = (int)std::thread::hardware_concurrency();
    uint64_t seed = 0xC0FFEEULL;
    bool check = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : "0"; };
        if (a == "--sims") sims = std::atol(val());
        else if (a == "--threads") threads = std::atoi(val());
        else if (a == "--seed") seed = std::strtoull(val(), nullptr, 10);
        else if (a == "--check") check = true;
        else if (a == "--help") {
            std::printf("usage: %s [--sims N] [--threads T] [--seed S] [--check]\n", argv[0]);
            return 0;
        }
    }
    if (threads < 1) threads = 1;
    if (sims < 1) sims = 1;

    const std::vector<Team> teams = default_bracket();
    const size_t n = teams.size();
    if (n < 2 || (n & (n - 1)) != 0) {
        std::fprintf(stderr, "bracket size must be a power of two\n");
        return 1;
    }
    const std::vector<double> w = build_win_matrix(teams);

    const long n_chunks = (sims + CHUNK - 1) / CHUNK;
    std::atomic<long> cursor{0};
    std::vector<long> totals(n, 0);
    std::mutex merge_mu;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    pool.reserve((size_t)threads);
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&]() {
            std::vector<long> local(n, 0);   // thread-private: no false sharing
            for (;;) {
                const long c = cursor.fetch_add(1, std::memory_order_relaxed);
                if (c >= n_chunks) break;
                const long todo = std::min<long>(CHUNK, sims - c * CHUNK);
                run_chunk(c, todo, w, n, seed, local);
            }
            std::lock_guard<std::mutex> lk(merge_mu);
            for (size_t i = 0; i < n; ++i) totals[i] += local[i];
        });
    }
    for (auto& th : pool) th.join();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::vector<double> exact;
    if (check) exact = exact_title_probs(teams);

    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return totals[a] > totals[b]; });

    std::printf("=== %ld simulations, %d threads, %.2fs (%.1fM sims/s) ===\n",
                sims, threads, secs, sims / secs / 1e6);
    if (check) std::printf("%3s %-14s %9s %9s %8s\n", "#", "team", "mc", "exact", "z");
    else       std::printf("%3s %-14s %9s\n", "#", "team", "mc");

    double max_z = 0.0;
    for (size_t k = 0; k < n; ++k) {
        const size_t i = order[k];
        const double p = (double)totals[i] / (double)sims;
        if (check) {
            const double se = std::sqrt(p * (1.0 - p) / (double)sims);
            const double z = se > 0 ? (p - exact[i]) / se : 0.0;
            max_z = std::max(max_z, std::fabs(z));
            std::printf("%3zu %-14s %8.4f%% %8.4f%% %8.2f\n",
                        k + 1, teams[i].name.c_str(), 100 * p, 100 * exact[i], z);
        } else {
            std::printf("%3zu %-14s %8.4f%%\n", k + 1, teams[i].name.c_str(), 100 * p);
        }
    }
    if (check) {
        std::printf("\nlargest |z| vs exact: %.2f  %s\n", max_z,
                    max_z < 4.0 ? "(consistent)" : "(SUSPICIOUS -- investigate)");
        std::printf("note: the exact column is O(n^2) and took microseconds. "
                    "MC is here for the models where no such recursion exists.\n");
    }
    return 0;
}
