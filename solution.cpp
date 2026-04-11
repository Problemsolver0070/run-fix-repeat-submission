#include <cstdio>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

const int MAXN = 125;
const int MAXNN = MAXN * MAXN;
const int NUM_LANDMARKS = 40;
const int THRESHOLD = 6;

int N, T;
char grid_raw[MAXN][MAXN + 2];
int grid[MAXNN]; // 1 = empty, 0 = wall
int empty_list[MAXNN];
int num_empty;
int empty_id[MAXNN]; // flat_idx -> compact index in empty_list (-1 if wall)

// Adjacency list for empty cells
vector<int> adj[MAXNN];

// BFS
int bfs_dist[MAXNN];
void bfs(int start) {
    memset(bfs_dist, -1, sizeof(int) * N * N);
    bfs_dist[start] = 0;
    queue<int> q;
    q.push(start);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        int d = bfs_dist[u] + 1;
        for (int v : adj[u]) {
            if (bfs_dist[v] < 0) {
                bfs_dist[v] = d;
                q.push(v);
            }
        }
    }
}

// Landmarks
int landmarks[NUM_LANDMARKS];
int num_landmarks;
// lm_dist[k][i] = distance from landmark k to empty_list[i]
int lm_dist[NUM_LANDMARKS][MAXNN]; // indexed by compact id

// Farthest-point sampling + store distances
int min_dist_to_lm[MAXNN]; // for farthest-point sampling, indexed by compact id

void select_landmarks_and_compute() {
    for (int i = 0; i < num_empty; i++) min_dist_to_lm[i] = 1000000000;

    num_landmarks = min(NUM_LANDMARKS, num_empty);

    // First landmark: pick cell closest to center
    int center = (N / 2) * N + (N / 2);
    int best_first = 0;
    int best_dist_to_center = 1000000000;
    for (int i = 0; i < num_empty; i++) {
        int idx = empty_list[i];
        int r = idx / N, c = idx % N;
        int d = abs(r - N/2) + abs(c - N/2);
        if (d < best_dist_to_center) {
            best_dist_to_center = d;
            best_first = i;
        }
    }

    landmarks[0] = empty_list[best_first];
    bfs(landmarks[0]);
    for (int i = 0; i < num_empty; i++) {
        lm_dist[0][i] = bfs_dist[empty_list[i]];
        if (lm_dist[0][i] >= 0 && lm_dist[0][i] < min_dist_to_lm[i])
            min_dist_to_lm[i] = lm_dist[0][i];
    }

    for (int lk = 1; lk < num_landmarks; lk++) {
        // Pick farthest empty cell from all existing landmarks
        int best_i = -1, best_val = -1;
        for (int i = 0; i < num_empty; i++) {
            if (min_dist_to_lm[i] > best_val) {
                best_val = min_dist_to_lm[i];
                best_i = i;
            }
        }
        if (best_i == -1) { num_landmarks = lk; break; }

        landmarks[lk] = empty_list[best_i];
        bfs(landmarks[lk]);
        for (int i = 0; i < num_empty; i++) {
            lm_dist[lk][i] = bfs_dist[empty_list[i]];
            if (lm_dist[lk][i] >= 0 && lm_dist[lk][i] < min_dist_to_lm[i])
                min_dist_to_lm[i] = lm_dist[lk][i];
        }
    }
}

// First landmark precomputation
int best_first_lm;
// bucket_map: distance -> list of compact indices
vector<int> bucket_map[MAXNN]; // index by distance value (max possible ~15000)
int max_dist_val;

// Second landmark precomputation
int second_lm_for_bucket[MAXNN]; // -1 if not needed
vector<int> sub_bucket_keys[MAXNN]; // for each first-lm distance, the possible second-lm distances
vector<int> sub_bucket_vals[MAXNN * 2]; // we'll use a different approach

// For sub-bucket: use a map-like structure
// sub_bucket[first_dist][second_dist] -> list of compact indices
// We'll flatten this: store for each first_dist bucket
struct SubBucket {
    int second_lm;
    vector<pair<int, vector<int>>> buckets; // (second_dist, candidate_list)
};
SubBucket sub_buckets[MAXNN];
bool has_sub_bucket[MAXNN];

void precompute_first_landmark() {
    best_first_lm = 0;
    int best_max_bucket = num_empty + 1;
    max_dist_val = 0;

    int counts[MAXNN];

    for (int k = 0; k < num_landmarks; k++) {
        memset(counts, 0, sizeof(int) * (max_dist_val + 1));
        int local_max_d = 0;
        int max_b = 0;
        for (int i = 0; i < num_empty; i++) {
            int d = lm_dist[k][i];
            if (d < 0) continue;
            if (d > local_max_d) local_max_d = d;
            counts[d]++;
            if (counts[d] > max_b) max_b = counts[d];
        }
        if (local_max_d > max_dist_val) max_dist_val = local_max_d;
        if (max_b < best_max_bucket) {
            best_max_bucket = max_b;
            best_first_lm = k;
        }
        // reset counts for used distances
        for (int i = 0; i < num_empty; i++) {
            int d = lm_dist[k][i];
            if (d >= 0) counts[d] = 0;
        }
    }

    // Build bucket_map for best_first_lm
    for (int i = 0; i <= max_dist_val; i++) bucket_map[i].clear();
    for (int i = 0; i < num_empty; i++) {
        int d = lm_dist[best_first_lm][i];
        if (d >= 0) bucket_map[d].push_back(i);
    }
}

void precompute_second_landmarks() {
    memset(has_sub_bucket, false, sizeof(has_sub_bucket));
    memset(second_lm_for_bucket, -1, sizeof(second_lm_for_bucket));

    int counts[MAXNN];

    for (int dv = 0; dv <= max_dist_val; dv++) {
        if ((int)bucket_map[dv].size() <= THRESHOLD) continue;

        const vector<int>& cands = bucket_map[dv];
        int best_k = -1, best_score = (int)cands.size() + 1;

        for (int k = 0; k < num_landmarks; k++) {
            if (k == best_first_lm) continue;

            int max_b = 0;
            bool aborted = false;
            // Use a temporary counter
            vector<pair<int,int>> used; // (dist, old_count) for cleanup
            for (int ci : cands) {
                int dd = lm_dist[k][ci];
                if (dd < 0) continue;
                counts[dd]++;
                used.push_back({dd, 0}); // just track which to reset
                if (counts[dd] > max_b) {
                    max_b = counts[dd];
                    if (max_b >= best_score) { aborted = true; break; }
                }
            }
            // Reset counts
            for (auto& p : used) counts[p.first] = 0;

            if (!aborted && max_b < best_score) {
                best_score = max_b;
                best_k = k;
            }
        }

        if (best_k != -1) {
            second_lm_for_bucket[dv] = best_k;
            has_sub_bucket[dv] = true;

            // Build sub-bucket map
            sub_buckets[dv].second_lm = best_k;
            sub_buckets[dv].buckets.clear();

            // Use counts to group
            // First pass: find all distinct distances
            vector<pair<int, vector<int>>> temp_map;
            // Use a simple approach: sort by second distance
            vector<pair<int,int>> dist_ci; // (second_dist, compact_idx)
            for (int ci : cands) {
                int dd = lm_dist[best_k][ci];
                dist_ci.push_back({dd, ci});
            }
            sort(dist_ci.begin(), dist_ci.end());

            int prev_d = -999;
            for (auto& p : dist_ci) {
                if (p.first != prev_d) {
                    temp_map.push_back({p.first, {}});
                    prev_d = p.first;
                }
                temp_map.back().second.push_back(p.second);
            }
            sub_buckets[dv].buckets = temp_map;
        }
    }

    memset(counts, 0, sizeof(counts));
}

// Query function
int query_count;
int query_cell(int flat_idx) {
    int r = flat_idx / N + 1;
    int c = flat_idx % N + 1;
    printf("%d %d\n", r, c);
    fflush(stdout);
    int resp;
    scanf("%d", &resp);
    query_count++;
    return resp;
}

// Select best landmark for a candidate list (dynamic, during hunting)
int select_best_landmark_dynamic(const vector<int>& cands, const vector<bool>& queried) {
    int best_k = -1;
    int best_score = (int)cands.size() + 1;

    for (int k = 0; k < num_landmarks; k++) {
        if (queried[k]) continue;

        // Count buckets
        int max_b = 0;
        // Use a hash-map-like approach with a small array
        // Since distances can be up to ~1000, use a temporary array
        static int tmp_counts[MAXNN];
        vector<int> used_dists;

        bool aborted = false;
        for (int ci : cands) {
            int dd = lm_dist[k][ci];
            if (dd < 0) continue;
            if (tmp_counts[dd] == 0) used_dists.push_back(dd);
            tmp_counts[dd]++;
            if (tmp_counts[dd] > max_b) {
                max_b = tmp_counts[dd];
                if (max_b >= best_score) { aborted = true; break; }
            }
        }

        // Reset
        for (int d : used_dists) tmp_counts[d] = 0;

        if (!aborted && max_b < best_score) {
            best_score = max_b;
            best_k = k;
        }
    }
    return best_k;
}

int main() {
    scanf("%d %d", &N, &T);
    int NN = N * N;

    memset(empty_id, -1, sizeof(empty_id));
    num_empty = 0;

    for (int r = 0; r < N; r++) {
        scanf("%s", grid_raw[r]);
        for (int c = 0; c < N; c++) {
            int idx = r * N + c;
            if (grid_raw[r][c] == '.') {
                grid[idx] = 1;
                empty_id[idx] = num_empty;
                empty_list[num_empty] = idx;
                num_empty++;
            } else {
                grid[idx] = 0;
            }
        }
    }

    // Build adjacency list
    for (int i = 0; i < num_empty; i++) {
        int idx = empty_list[i];
        int r = idx / N, c = idx % N;
        if (r > 0 && grid[idx - N]) adj[idx].push_back(idx - N);
        if (r < N - 1 && grid[idx + N]) adj[idx].push_back(idx + N);
        if (c > 0 && grid[idx - 1]) adj[idx].push_back(idx - 1);
        if (c < N - 1 && grid[idx + 1]) adj[idx].push_back(idx + 1);
    }

    // Select landmarks and compute distances
    select_landmarks_and_compute();

    // Precompute first landmark
    precompute_first_landmark();

    // Precompute second landmarks
    precompute_second_landmarks();

    // Hunting loop
    query_count = 0;

    while (query_count < T) {
        // Query first landmark
        int resp = query_cell(landmarks[best_first_lm]);
        if (resp == 0) continue;
        if (resp < 0) continue;

        int d1 = resp;
        if (d1 > max_dist_val || bucket_map[d1].empty()) continue;

        vector<int> candidates = bucket_map[d1]; // copy
        vector<bool> queried_lms(num_landmarks, false);
        queried_lms[best_first_lm] = true;

        // Use precomputed second landmark if available
        if (has_sub_bucket[d1] && query_count < T) {
            int k2 = sub_buckets[d1].second_lm;
            int resp2 = query_cell(landmarks[k2]);
            queried_lms[k2] = true;

            if (resp2 == 0) continue;

            // Find in sub-bucket
            candidates.clear();
            for (auto& p : sub_buckets[d1].buckets) {
                if (p.first == resp2) {
                    candidates = p.second;
                    break;
                }
            }
            if (candidates.empty()) continue;
        }

        // Continue narrowing with dynamic landmark selection
        while ((int)candidates.size() > THRESHOLD && query_count < T) {
            int k = select_best_landmark_dynamic(candidates, queried_lms);
            if (k == -1) break;

            int resp_k = query_cell(landmarks[k]);
            queried_lms[k] = true;

            if (resp_k == 0) { candidates.clear(); break; }

            vector<int> new_cands;
            for (int ci : candidates) {
                if (lm_dist[k][ci] == resp_k) {
                    new_cands.push_back(ci);
                }
            }
            candidates = new_cands;
        }

        if (candidates.empty()) continue;

        // Query remaining candidates directly
        for (int ci : candidates) {
            if (query_count >= T) break;
            int resp_c = query_cell(empty_list[ci]);
            if (resp_c == 0) break;
        }
    }

    return 0;
}
