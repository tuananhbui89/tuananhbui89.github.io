// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <set>
#include <map>
#include <chrono>
#include <random>
#include <unordered_map>

// Timer
std::chrono::steady_clock::time_point program_start_time;
std::chrono::milliseconds time_limit_ms(1850); 

// Global problem parameters
int N_items_global, D_groups_global, Q_total_global;
int queries_made = 0;

std::mt19937 rng_engine;

// Query Manager with optimized caching
class QueryManager {
private:
    int N, Q;
    int& queries_made_ref;
    std::vector<char> cmp1_flat; // flat N*N storage for 1v1 comparisons
    std::unordered_map<uint32_t, char> cmp1v2; // for 1v2 comparisons
    std::mt19937& rng;

    inline uint32_t key1v2(int a, int b, int c) const {
        int mn = std::min(b, c), mx = std::max(b, c);
        return (static_cast<uint32_t>(a) << 16) | (static_cast<uint32_t>(mn) << 8) | static_cast<uint32_t>(mx);
    }

    char perform_query_actual(const std::vector<int>& L_items, const std::vector<int>& R_items) {
        queries_made_ref++;
        std::cout << L_items.size() << " " << R_items.size();
        for (int item_idx : L_items) {
            std::cout << " " << item_idx;
        }
        for (int item_idx : R_items) {
            std::cout << " " << item_idx;
        }
        std::cout << std::endl;

        char result_char;
        std::cin >> result_char;
        return result_char;
    }

public:
    QueryManager(int N_, int Q_, int& qm, std::mt19937& r) : N(N_), Q(Q_), queries_made_ref(qm), rng(r) {
        cmp1_flat.assign(N * N, 0);
        cmp1v2.reserve(N * N / 4 + 10);
    }

    char compare1(int a, int b) {
        if (a == b) return '=';
        int mn = std::min(a, b), mx = std::max(a, b);
        char cached = cmp1_flat[mn * N + mx];
        if (cached != 0) {
            if (a == mn) return cached;
            return (cached == '<' ? '>' : (cached == '>' ? '<' : '='));
        }
        if (queries_made_ref >= Q) return '=';
        
        char res = perform_query_actual({a}, {b});
        if (a == mn) {
            cmp1_flat[mn * N + mx] = res;
        } else {
            if (res == '<') cmp1_flat[mn * N + mx] = '>';
            else if (res == '>') cmp1_flat[mn * N + mx] = '<';
            else cmp1_flat[mn * N + mx] = '=';
        }
        return res;
    }

    char compare1v2(int item_curr, int item_prev, int item_s_aux) {
        if (item_curr == item_prev || item_curr == item_s_aux || item_prev == item_s_aux) {
            if (item_prev == item_s_aux) return compare1(item_curr, item_prev);
            if (item_curr == item_prev) return compare1(item_curr, item_s_aux);
            return compare1(item_curr, item_prev);
        }
        uint32_t key = key1v2(item_curr, item_prev, item_s_aux);
        auto it = cmp1v2.find(key);
        if (it != cmp1v2.end()) return it->second;
        if (queries_made_ref >= Q) return '=';
        char res = perform_query_actual({item_curr}, {item_prev, item_s_aux});
        cmp1v2.emplace(key, res);
        return res;
    }

    void exhaust_queries() {
        if (N >= 2) {
            int a = 0, b = 1;
            while (queries_made_ref < Q) {
                perform_query_actual({a}, {b});
                ++b;
                if (b == a) ++b;
                if (b >= N) { 
                    b = 0; 
                    a = (a + 1) % N; 
                    if (b == a) b = (b + 1) % N; 
                }
            }
        }
    }
};

// Weight estimation module
class WeightEstimator {
private:
    static constexpr long long BASE_WEIGHT = 100000;
    static constexpr int FACTOR_GT = 200;
    static constexpr int FACTOR_LT = 50;
    static constexpr int FACTOR_XJ_FALLBACK = 100;

    QueryManager& qm;
    int N, D, Q;

    double estimate_log2(double val) {
        return (val <= 1.0) ? 0.0 : std::log2(val);
    }

    int calculate_query_cost(int N_val, int k_pivots) {
        if (k_pivots <= 0) return 0;
        if (k_pivots == 1) return std::max(0, N_val - 1);
        double cost = 0;
        cost += k_pivots * estimate_log2(k_pivots);
        for (int j = 2; j < k_pivots; ++j) {
            if (j - 1 > 0) cost += estimate_log2(j - 1);
        }
        cost += (N_val - k_pivots) * estimate_log2(k_pivots);
        return static_cast<int>(std::ceil(cost));
    }

    void merge_sort_pivots(std::vector<int>& pivots, int left, int right) {
        if (left >= right) return;
        int mid = (left + right) / 2;
        merge_sort_pivots(pivots, left, mid);
        merge_sort_pivots(pivots, mid + 1, right);
        
        int n1 = mid - left + 1, n2 = right - mid;
        std::vector<int> L(n1), R(n2);
        for (int i = 0; i < n1; ++i) L[i] = pivots[left + i];
        for (int j = 0; j < n2; ++j) R[j] = pivots[mid + 1 + j];
        
        int i = 0, j = 0, k = left;
        while (i < n1 && j < n2) {
            char cmp = qm.compare1(L[i], R[j]);
            if (cmp == '<' || cmp == '=') pivots[k++] = L[i++];
            else pivots[k++] = R[j++];
        }
        while (i < n1) pivots[k++] = L[i++];
        while (j < n2) pivots[k++] = R[j++];
    }

public:
    WeightEstimator(QueryManager& qm_, int N_, int D_, int Q_) : qm(qm_), N(N_), D(D_), Q(Q_) {}

    std::vector<long long> estimate_weights() {
        std::vector<long long> weights(N, BASE_WEIGHT);
        
        // Determine pivot count
        int k_pivots = (N > 0) ? 1 : 0;
        if (N > 1) {
            for (int k = N; k >= 1; --k) {
                if (calculate_query_cost(N, k) <= Q) {
                    k_pivots = k;
                    break;
                }
            }
        }
        k_pivots = std::min(k_pivots, N);

        if (k_pivots == 0) return weights;

        // Select and sort pivots
        std::vector<int> pivots(k_pivots);
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_engine);
        for (int i = 0; i < k_pivots; ++i) pivots[i] = indices[i];

        if (k_pivots >= 2) {
            merge_sort_pivots(pivots, 0, k_pivots - 1);
        }

        // Estimate pivot weights
        if (k_pivots == 1) {
            weights[pivots[0]] = BASE_WEIGHT;
            for (int i = 0; i < N; ++i) {
                if (i == pivots[0]) continue;
                char res = qm.compare1(i, pivots[0]);
                if (res == '=') weights[i] = BASE_WEIGHT;
                else if (res == '<') weights[i] = std::max(1LL, BASE_WEIGHT * FACTOR_LT / 100);
                else weights[i] = std::max(1LL, BASE_WEIGHT * FACTOR_GT / 100);
            }
        } else {
            // Multi-pivot estimation
            weights[pivots[0]] = BASE_WEIGHT;
            
            // Handle p1
            char res_p1 = qm.compare1(pivots[1], pivots[0]);
            if (res_p1 == '=') weights[pivots[1]] = weights[pivots[0]];
            else if (res_p1 == '<') weights[pivots[1]] = std::max(1LL, weights[pivots[0]] * FACTOR_LT / 100);
            else weights[pivots[1]] = std::max(1LL, weights[pivots[0]] * FACTOR_GT / 100);
            
            if (res_p1 == '>' && weights[pivots[1]] == weights[pivots[0]]) {
                weights[pivots[1]] = weights[pivots[0]] + 1;
            }

            // Handle remaining pivots with binary search bracketing
            long long max_bound = BASE_WEIGHT * (N / std::max(1, D) + 10);
            for (int j = 2; j < k_pivots; ++j) {
                int cur = pivots[j], prev = pivots[j-1];
                char res = qm.compare1(cur, prev);
                
                if (res == '=') {
                    weights[cur] = weights[prev];
                } else if (res == '<') {
                    weights[cur] = std::max(1LL, weights[prev] * FACTOR_LT / 100);
                } else {
                    // Binary search to bracket X_j
                    long long X_low = 1, X_high = max_bound;
                    bool low_set = false, high_set = false;
                    
                    int low_idx = 0, high_idx = j - 2;
                    int tries = std::max(1, static_cast<int>(std::ceil(estimate_log2(std::max(1, high_idx - low_idx + 1)))));
                    
                    for (int t = 0; t < tries && low_idx <= high_idx && queries_made < Q; ++t) {
                        int mid_idx = (low_idx + high_idx) / 2;
                        int s = pivots[mid_idx];
                        char res_1v2 = qm.compare1v2(cur, prev, s);
                        
                        if (res_1v2 == '=') {
                            X_low = X_high = weights[s];
                            low_set = high_set = true;
                            break;
                        } else if (res_1v2 == '<') {
                            X_high = weights[s];
                            high_set = true;
                            high_idx = mid_idx - 1;
                        } else {
                            X_low = weights[s];
                            low_set = true;
                            low_idx = mid_idx + 1;
                        }
                    }
                    
                    long long est_X;
                    if (low_set && !high_set) est_X = X_low * FACTOR_GT / 100;
                    else if (!low_set && high_set) est_X = X_high * FACTOR_LT / 100;
                    else if (low_set && high_set) est_X = (X_low + X_high) / 2;
                    else est_X = weights[prev] * FACTOR_XJ_FALLBACK / 100;
                    
                    est_X = std::max(1LL, est_X);
                    weights[cur] = weights[prev] + est_X;
                }
                
                // Ensure monotonicity
                if (weights[cur] < weights[prev]) weights[cur] = weights[prev];
                if (res == '>' && weights[cur] == weights[prev]) weights[cur] = weights[prev] + 1;
            }

            // Estimate non-pivot weights
            std::vector<bool> is_pivot(N, false);
            for (int p : pivots) is_pivot[p] = true;
            
            for (int i = 0; i < N; ++i) {
                if (is_pivot[i]) continue;
                
                int low = 0, high = k_pivots - 1, found = -1;
                while (low <= high && queries_made < Q) {
                    int mid = (low + high) / 2;
                    char res = qm.compare1(i, pivots[mid]);
                    if (res == '=') { found = mid; break; }
                    else if (res == '<') high = mid - 1;
                    else low = mid + 1;
                }
                
                if (found != -1) {
                    weights[i] = weights[pivots[found]];
                    continue;
                }
                
                int pos = low;
                if (pos == 0) {
                    long long w0 = weights[pivots[0]];
                    if (k_pivots >= 2) {
                        long long w1 = weights[pivots[1]];
                        if (w1 > w0 && w0 > 0) weights[i] = std::max(1LL, w0 * w0 / w1);
                        else weights[i] = std::max(1LL, w0 / 2);
                    } else {
                        weights[i] = std::max(1LL, w0 / 2);
                    }
                } else if (pos == k_pivots) {
                    long long wk1 = weights[pivots[k_pivots - 1]];
                    if (k_pivots >= 2) {
                        long long wk2 = weights[pivots[k_pivots - 2]];
                        if (wk1 > wk2 && wk2 > 0) weights[i] = std::max(1LL, wk1 * wk1 / wk2);
                        else weights[i] = std::max(1LL, wk1 * 2);
                    } else {
                        weights[i] = std::max(1LL, wk1 * 2);
                    }
                } else {
                    long long wl = weights[pivots[pos - 1]];
                    long long wr = weights[pivots[pos]];
                    if (wl > 0 && wr > 0) {
                        weights[i] = static_cast<long long>(std::sqrt(static_cast<double>(wl) * wr));
                    } else {
                        weights[i] = (wl + wr) / 2;
                    }
                    weights[i] = std::max(weights[i], wl);
                    weights[i] = std::min(weights[i], wr);
                }
                weights[i] = std::max(1LL, weights[i]);
            }
        }

        // Final validation
        for (int i = 0; i < N; ++i) {
            if (weights[i] <= 0) weights[i] = BASE_WEIGHT;
        }

        return weights;
    }
};

// Assignment optimizer
class AssignmentOptimizer {
private:
    int N, D;
    std::vector<long long>& weights;
    std::mt19937& rng;
    
    double calc_variance(const std::vector<long long>& sums, long long total) {
        if (D <= 0) return 1e18;
        double mean = static_cast<double>(total) / D;
        double sum_sq = 0;
        for (long long s : sums) sum_sq += static_cast<double>(s) * s;
        double var = sum_sq / D - mean * mean;
        return std::max(0.0, var);
    }

public:
    AssignmentOptimizer(int N_, int D_, std::vector<long long>& w, std::mt19937& r) 
        : N(N_), D(D_), weights(w), rng(r) {}

    std::vector<int> optimize() {
        std::vector<int> assignment(N, 0);
        std::vector<long long> group_sums(D, 0);
        std::vector<std::vector<int>> group_items(D);
        std::vector<int> item_pos(N);
        
        // Greedy initialization
        std::vector<std::pair<long long, int>> sorted_items;
        for (int i = 0; i < N; ++i) {
            sorted_items.emplace_back(-weights[i], i);
        }
        std::sort(sorted_items.begin(), sorted_items.end());
        
        long long total_sum = 0;
        for (auto [neg_w, item] : sorted_items) {
            int best_group = 0;
            for (int g = 1; g < D; ++g) {
                if (group_sums[g] < group_sums[best_group]) best_group = g;
            }
            assignment[item] = best_group;
            item_pos[item] = group_items[best_group].size();
            group_items[best_group].push_back(item);
            group_sums[best_group] += weights[item];
            total_sum += weights[item];
        }

        double current_var = calc_variance(group_sums, total_sum);

        // Enhanced local search with best-of-K
        if (D > 1) {
            const int MAX_ITERS = 400;
            const int K_ITEMS = 8;
            
            for (int iter = 0; iter < MAX_ITERS; ++iter) {
                if ((iter & 31) == 0) {
                    auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - program_start_time) >= time_limit_ms) break;
                }
                
                int max_g = 0, min_g = 0;
                for (int g = 1; g < D; ++g) {
                    if (group_sums[g] > group_sums[max_g]) max_g = g;
                    if (group_sums[g] < group_sums[min_g]) min_g = g;
                }
                if (max_g == min_g || group_items[max_g].empty()) break;

                // Find best relocate from max_g to min_g among top-K heaviest
                std::vector<std::pair<long long, int>> candidates;
                for (int item : group_items[max_g]) {
                    candidates.emplace_back(weights[item], item);
                }
                if (candidates.empty()) break;
                std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                if ((int)candidates.size() > K_ITEMS) candidates.resize(K_ITEMS);

                double best_var = current_var;
                int best_item = -1;
                for (auto [w, item] : candidates) {
                    long long new_max = group_sums[max_g] - w;
                    long long new_min = group_sums[min_g] + w;
                    double new_var = calc_variance({new_max, new_min}, group_sums[max_g] + group_sums[min_g]);
                    if (new_var + 1e-12 < best_var) {
                        best_var = new_var;
                        best_item = item;
                    }
                }

                if (best_item == -1) break;

                // Apply move
                long long w = weights[best_item];
                group_sums[max_g] -= w;
                group_sums[min_g] += w;
                current_var = calc_variance(group_sums, total_sum);

                // Update tracking
                int pos = item_pos[best_item];
                int last = group_items[max_g].back();
                if (best_item != last) {
                    group_items[max_g][pos] = last;
                    item_pos[last] = pos;
                }
                group_items[max_g].pop_back();
                item_pos[best_item] = group_items[min_g].size();
                group_items[min_g].push_back(best_item);
                assignment[best_item] = min_g;
            }
        }

        // Targeted Simulated Annealing
        if (D > 1) {
            double T = std::max(1.0, current_var * 0.25);
            double cool_rate = 0.99985;
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            int iterations = 0, no_imp = 0;
            
            while (true) {
                ++iterations;
                if ((iterations & 255) == 0) {
                    auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - program_start_time) >= time_limit_ms) break;
                    T *= cool_rate;
                    if (T < 1e-12) break;
                }

                // Targeted moves: 75% heavy-to-light relocate, 25% swap
                if ((rng() % 4) != 0) {
                    // Targeted relocate
                    int max_g = 0, min_g = 0;
                    for (int g = 1; g < D; ++g) {
                        if (group_sums[g] > group_sums[max_g]) max_g = g;
                        if (group_sums[g] < group_sums[min_g]) min_g = g;
                    }
                    
                    if (group_items[max_g].empty()) { ++no_imp; continue; }
                    
                    // Pick heavy item from max group (best of 3 samples)
                    int item = group_items[max_g][rng() % group_items[max_g].size()];
                    for (int s = 0; s < 2; ++s) {
                        int cand = group_items[max_g][rng() % group_items[max_g].size()];
                        if (weights[cand] > weights[item]) item = cand;
                    }
                    
                    long long w = weights[item];
                    long long new_max = group_sums[max_g] - w;
                    long long new_min = group_sums[min_g] + w;
                    
                    double new_var = current_var;
                    new_var -= (static_cast<double>(group_sums[max_g]) * group_sums[max_g]) / D;
                    new_var -= (static_cast<double>(group_sums[min_g]) * group_sums[min_g]) / D;
                    new_var += (static_cast<double>(new_max) * new_max) / D;
                    new_var += (static_cast<double>(new_min) * new_min) / D;
                    
                    double delta = new_var - current_var;
                    if (delta < 0 || unif(rng) < std::exp(-delta / T)) {
                        // Accept move
                        current_var = new_var;
                        group_sums[max_g] = new_max;
                        group_sums[min_g] = new_min;
                        
                        int pos = item_pos[item];
                        int last = group_items[max_g].back();
                        if (item != last) {
                            group_items[max_g][pos] = last;
                            item_pos[last] = pos;
                        }
                        group_items[max_g].pop_back();
                        item_pos[item] = group_items[min_g].size();
                        group_items[min_g].push_back(item);
                        assignment[item] = min_g;
                        
                        if (delta < -1e-12) no_imp = 0; else ++no_imp;
                    } else ++no_imp;
                } else {
                    // Random swap
                    int g1 = rng() % D, g2 = rng() % D;
                    while (g2 == g1) g2 = rng() % D;
                    if (group_items[g1].empty() || group_items[g2].empty()) { ++no_imp; continue; }
                    
                    int a = group_items[g1][rng() % group_items[g1].size()];
                    int b = group_items[g2][rng() % group_items[g2].size()];
                    long long wa = weights[a], wb = weights[b];
                    
                    long long new_g1 = group_sums[g1] - wa + wb;
                    long long new_g2 = group_sums[g2] - wb + wa;
                    
                    double new_var = current_var;
                    new_var -= (static_cast<double>(group_sums[g1]) * group_sums[g1]) / D;
                    new_var -= (static_cast<double>(group_sums[g2]) * group_sums[g2]) / D;
                    new_var += (static_cast<double>(new_g1) * new_g1) / D;
                    new_var += (static_cast<double>(new_g2) * new_g2) / D;
                    
                    double delta = new_var - current_var;
                    if (delta < 0 || unif(rng) < std::exp(-delta / T)) {
                        current_var = new_var;
                        group_sums[g1] = new_g1;
                        group_sums[g2] = new_g2;
                        
                        // Swap items
                        int pos_a = item_pos[a], pos_b = item_pos[b];
                        int back_a = group_items[g1].back(), back_b = group_items[g2].back();
                        if (a != back_a) { group_items[g1][pos_a] = back_a; item_pos[back_a] = pos_a; }
                        group_items[g1].pop_back();
                        if (b != back_b) { group_items[g2][pos_b] = back_b; item_pos[back_b] = pos_b; }
                        group_items[g2].pop_back();
                        
                        item_pos[b] = group_items[g1].size(); group_items[g1].push_back(b); assignment[b] = g1;
                        item_pos[a] = group_items[g2].size(); group_items[g2].push_back(a); assignment[a] = g2;
                        
                        if (delta < -1e-12) no_imp = 0; else ++no_imp;
                    } else ++no_imp;
                }
                
                if (no_imp > N * 12) break;
            }
        }

        return assignment;
    }
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    program_start_time = std::chrono::steady_clock::now();
    uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    rng_engine.seed(seed);

    std::cin >> N_items_global >> D_groups_global >> Q_total_global;

    QueryManager qm(N_items_global, Q_total_global, queries_made, rng_engine);
    WeightEstimator estimator(qm, N_items_global, D_groups_global, Q_total_global);
    
    std::vector<long long> weights = estimator.estimate_weights();
    
    qm.exhaust_queries();
    
    AssignmentOptimizer optimizer(N_items_global, D_groups_global, weights, rng_engine);
    std::vector<int> assignment = optimizer.optimize();

    for (int i = 0; i < N_items_global; ++i) {
        std::cout << assignment[i] << (i + 1 == N_items_global ? '\n' : ' ');
    }

    return 0;
}
// EVOLVE-BLOCK-END