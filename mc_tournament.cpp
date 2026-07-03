// Fast Monte Carlo simulation of a World Cup knockout bracket.
//
// Each tie is decided by an Elo win probability; we simulate the whole
// single-elimination bracket many times and estimate each team's title
// probability. This is the performance-critical companion to wc_model.py:
// 10M simulations run in well under a second thanks to a tight loop and a
// fast xorshift RNG.
//
// Build:  g++ -O2 -std=c++17 mc_tournament.cpp -o mc_tournament
// Run:    ./mc_tournament [num_simulations]

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

struct Team { std::string name; double elo; };

// A 16-team bracket (seeded in bracket order). Elo ratings are illustrative.
static std::vector<Team> bracket = {
    {"Brazil", 2030},   {"Serbia", 1780},
    {"France", 1990},   {"Denmark", 1850},
    {"Argentina", 2010},{"Mexico", 1820},
    {"Portugal", 1975}, {"Uruguay", 1890},
    {"Spain", 1980},    {"Morocco", 1810},
    {"Germany", 1960},  {"Japan", 1830},
    {"England", 1955},  {"Senegal", 1800},
    {"Netherlands", 1945}, {"USA", 1790},
};

// xorshift128+ : small, fast, good enough for Monte Carlo.
struct Rng {
    uint64_t s0, s1;
    explicit Rng(uint64_t seed) : s0(seed ^ 0x9E3779B97F4A7C15ULL), s1(seed * 2685821657736338717ULL + 1) {}
    inline uint64_t next() {
        uint64_t x = s0, y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s1 + y;
    }
    inline double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

inline double elo_win_prob(double a, double b) {
    return 1.0 / (1.0 + std::pow(10.0, (b - a) / 400.0));
}

int main(int argc, char** argv) {
    const long sims = (argc > 1) ? std::atol(argv[1]) : 10'000'000L;
    const int n = static_cast<int>(bracket.size());

    std::vector<long> titles(n, 0);
    std::vector<int> alive(n);
    Rng rng(88172645463325252ULL);

    for (long s = 0; s < sims; ++s) {
        for (int i = 0; i < n; ++i) alive[i] = i;   // reset bracket

        int remaining = n;
        while (remaining > 1) {
            for (int i = 0; i < remaining / 2; ++i) {
                int a = alive[2 * i], b = alive[2 * i + 1];
                double pa = elo_win_prob(bracket[a].elo, bracket[b].elo);
                alive[i] = (rng.uniform() < pa) ? a : b;   // winner advances
            }
            remaining /= 2;
        }
        titles[alive[0]]++;
    }

    std::printf("=== Championship probabilities over %ld simulations ===\n", sims);
    // simple selection sort by probability for a clean ranked print
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (titles[order[j]] > titles[order[i]]) std::swap(order[i], order[j]);

    for (int k = 0; k < n; ++k) {
        int t = order[k];
        std::printf("%2d. %-12s %6.2f%%\n", k + 1, bracket[t].name.c_str(),
                    100.0 * titles[t] / sims);
    }
    return 0;
}
