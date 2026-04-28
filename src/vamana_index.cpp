#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ============================================================================
// Greedy Search
// ============================================================================
// Beam search from a caller-supplied entry node.  Maintains a candidate set
// of at most L nodes ordered by distance, always expanding the closest
// unvisited node until no unexpanded candidates remain.
//
// The entry point is now a parameter so that PC Teleport can supply a
// query-specific starting node instead of the fixed start_node_.

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L,
                           uint32_t entry) const {
    std::set<Candidate> candidate_set;
    std::vector<bool>   visited(npts_, false);
    std::set<uint32_t>  expanded;
    uint32_t dist_cmps = 0;

    // Seed with the supplied entry point.
    float start_dist = compute_l2sq(query, get_vector(entry), dim_);
    dist_cmps++;
    candidate_set.insert({start_dist, entry});
    visited[entry] = true;

    while (true) {
        // Find the closest candidate that has not been expanded yet.
        uint32_t best_node = UINT32_MAX;
        for (const auto& [dist, id] : candidate_set) {
            if (!expanded.count(id)) {
                best_node = id;
                break;
            }
        }
        if (best_node == UINT32_MAX)
            break;

        expanded.insert(best_node);

        // Copy neighbor list under lock to avoid data races with parallel build.
        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }

        for (uint32_t nbr : neighbors) {
            if (visited[nbr]) continue;
            visited[nbr] = true;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            if (candidate_set.size() < L) {
                candidate_set.insert({d, nbr});
            } else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nbr});
                }
            }
        }
    }

    std::vector<Candidate> results(candidate_set.begin(), candidate_set.end());
    return {results, dist_cmps};
}

// ============================================================================
// Robust Prune  —  Standard Alpha-RNG
// ============================================================================
// Prunes candidates to at most R diverse neighbors using the alpha-RNG rule.
// A candidate c is kept only if dist(node, c) <= alpha * dist(c, n) for all
// already-selected neighbors n.  The supplied alpha is used as-is for every
// node (no per-node adaptation).

void VamanaIndex::robust_prune(uint32_t node,
                               std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    // Remove self from candidates.
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    // Sort by ascending distance to node.
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R) break;

        // Keep this candidate iff it is not "occluded" by any already-chosen
        // neighbor: occluded means  dist(node, cand) > alpha * dist(cand, n).
        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > alpha * dist_cand_to_selected) {
                keep = false;
                break;
            }
        }
        if (keep)
            new_neighbors.push_back(cand_id);
    }

    graph_[node] = std::move(new_neighbors);
}

// ============================================================================
// PC Teleport — compute_pca
// ============================================================================
// Computes the dataset centroid and the first principal component (PC1) via
// power iteration, then pre-computes every point's scalar projection onto
// PC1 and sorts the list.  Total cost: O(iters * npts * dim).
//
// At query time, get_entry_point() binary-searches this sorted list in
// O(log npts) to find the dataset point whose PC1 projection is nearest to
// the query's.  That point becomes the greedy-search entry — much better
// initialisation than a random node, with essentially zero per-query overhead.

void VamanaIndex::compute_pca(int iters) {
    // ---- Centroid ----
    mean_vec_.assign(dim_, 0.0f);
    for (uint32_t i = 0; i < npts_; ++i) {
        const float* v = row(i);
        for (uint32_t d = 0; d < dim_; ++d)
            mean_vec_[d] += v[d];
    }
    for (float& val : mean_vec_) val /= static_cast<float>(npts_);

    // ---- Power iteration for PC1 ----
    // Start from a constant vector; it converges in ~10-20 steps for typical
    // ANN datasets.
    pc_vec_.assign(dim_, 1.0f / std::sqrt(static_cast<float>(dim_)));

    std::vector<float> next_pc(dim_);
    for (int iter = 0; iter < iters; ++iter) {
        std::fill(next_pc.begin(), next_pc.end(), 0.0f);

        // next_pc = (X - mean)^T (X - mean) pc_vec  — one step of power method.
        for (uint32_t i = 0; i < npts_; ++i) {
            const float* v = row(i);
            float dot = 0.0f;
            for (uint32_t d = 0; d < dim_; ++d)
                dot += (v[d] - mean_vec_[d]) * pc_vec_[d];
            for (uint32_t d = 0; d < dim_; ++d)
                next_pc[d] += dot * (v[d] - mean_vec_[d]);
        }

        // Normalize.
        float norm = 0.0f;
        for (float val : next_pc) norm += val * val;
        norm = std::sqrt(norm);
        if (norm < 1e-12f) break;  // degenerate dataset — stop early
        for (uint32_t d = 0; d < dim_; ++d)
            pc_vec_[d] = next_pc[d] / norm;
    }

    // ---- Pre-compute sorted projections ----
    projections_.resize(npts_);
    for (uint32_t i = 0; i < npts_; ++i) {
        const float* v = row(i);
        float proj = 0.0f;
        for (uint32_t d = 0; d < dim_; ++d)
            proj += (v[d] - mean_vec_[d]) * pc_vec_[d];
        projections_[i] = {proj, i};
    }
    std::sort(projections_.begin(), projections_.end());
}

// ============================================================================
// PC Teleport — get_entry_point
// ============================================================================
// Projects the query onto PC1 then binary-searches the sorted projections
// array for the nearest value.  Returns the corresponding point id.

uint32_t VamanaIndex::get_entry_point(const float* query) const {
    float q_proj = 0.0f;
    for (uint32_t d = 0; d < dim_; ++d)
        q_proj += (query[d] - mean_vec_[d]) * pc_vec_[d];

    // Find the first projection >= q_proj.
    auto it = std::lower_bound(
        projections_.begin(), projections_.end(),
        std::make_pair(q_proj, 0u));

    if (it == projections_.end())
        return projections_.back().second;

    if (it == projections_.begin())
        return it->second;

    // Compare the two surrounding entries and pick the closer one.
    auto prev_it = std::prev(it);
    if (std::abs(prev_it->first - q_proj) < std::abs(it->first - q_proj))
        return prev_it->second;

    return it->second;
}

// ============================================================================
// Build Pass  —  internal helper
// ============================================================================
// A single insertion pass over the permutation `perm`.
// If clear_graph is true, all adjacency lists are zeroed before the pass
// (used for pass 1).  Pass 2 starts from the graph left by pass 1.

void VamanaIndex::build_pass(const std::vector<uint32_t>& perm,
                             uint32_t R, uint32_t L,
                             float pass_alpha, float gamma,
                             bool clear_graph) {
    if (clear_graph) {
        for (auto& adj : graph_) adj.clear();
    }

    const uint32_t gamma_R = static_cast<uint32_t>(gamma * R);

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        // Step 1 (pass 2 only): snapshot pass-1 neighbors BEFORE searching.
        //
        // We read graph_[point] now, under its lock, before greedy_search
        // runs.  This is safe because no other thread writes graph_[point]'s
        // forward edges — only backward-edge insertion (step 4 below) touches
        // it under the lock, and we haven't started that yet for this point.
        // Reading here avoids any race with the write in robust_prune (step 3)
        // which runs unlocked because this thread is the sole forward-edge
        // owner of `point` in the permutation.
        std::vector<uint32_t> pass1_nbrs;
        if (!clear_graph) {
            std::lock_guard<std::mutex> lk(locks_[point]);
            pass1_nbrs = graph_[point];
        }

        // Step 2: Greedy search to collect candidates.
        //
        // Always use start_node_ as the build-time entry for both passes.
        // PC Teleport is a query-time optimisation only — using it during
        // build concentrates traffic on PC1-median nodes, over-prunes their
        // neighborhoods, and hurts recall.
        auto [candidates, _dist_cmps] =
            greedy_search(get_vector(point), L, start_node_);

        // Step 3 (pass 2 only): merge pass-1 neighbors into candidates.
        //
        // robust_prune overwrites graph_[point] from scratch using only what
        // greedy_search found.  If L is too small, close pass-1 neighbors
        // not rediscovered by the search are permanently lost.  Merging them
        // in ensures pass 2 refines the graph rather than replacing it.
        // pass1_nbrs was captured before the search so there is no race.
        for (uint32_t existing_nbr : pass1_nbrs) {
            float d = compute_l2sq(row(point), row(existing_nbr), dim_);
            candidates.push_back({d, existing_nbr});
        }

        // Step 4: Prune candidates → write this node's forward edges.
        // No lock needed: each point appears exactly once in perm so only
        // this thread writes graph_[point]'s forward list.
        robust_prune(point, candidates, pass_alpha, R);

        // graph_[point] is now stable for this thread — read it directly.
        // (No other thread writes forward edges for `point`; backward-edge
        // writers hold locks_[point] but only append/prune, and we finished
        // our forward write above.)
        const std::vector<uint32_t> fwd_nbrs = graph_[point];

        // Step 4: Add backward edges from each forward neighbor back to point.
        for (uint32_t nbr : fwd_nbrs) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);

            // Avoid duplicate backward edges in pass 2.
            bool already_present = false;
            for (uint32_t existing : graph_[nbr]) {
                if (existing == point) { already_present = true; break; }
            }
            if (!already_present)
                graph_[nbr].push_back(point);

            // Step 5: If nbr's degree exceeds gamma*R, prune it.
            if (graph_[nbr].size() > gamma_R) {
                std::vector<Candidate> nbr_cands;
                nbr_cands.reserve(graph_[nbr].size());
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(row(nbr), row(nn), dim_);
                    nbr_cands.push_back({d, nn});
                }
                robust_prune(nbr, nbr_cands, pass_alpha, R);
            }
        }

        if (idx % 10000 == 0) {
            #pragma omp critical
            std::cout << "\r  Inserted " << idx << " / " << npts_
                      << " points" << std::flush;
        }
    }
    std::cout << "\r  Inserted " << npts_ << " / " << npts_
              << " points" << std::endl;
}

// ============================================================================
// Build  —  Two-Pass Construction
// ============================================================================
// Overview of the two enhancements and how they interleave:
//
//  [PRE-PASS]
//    • compute_pca()  — establishes PC1 projections for PC Teleport.
//
//  [PASS 1]   pass_alpha = 1.0
//    • Local, short-range graph built quickly.
//
//  [PASS 2]   pass_alpha = user alpha (>= 1.0)
//    • Refines the graph starting from pass-1 edges.
//    • robust_prune uses the fixed user alpha uniformly across all nodes.

void VamanaIndex::build(const std::string& data_path,
                        uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    // ---- Load data ----
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;
    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    if (L < R) {
        std::cerr << "Warning: L (" << L << ") < R (" << R
                  << "). Setting L = R." << std::endl;
        L = R;
    }

    // ---- Initialise graph and locks ----
    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // ---- Fixed start node (fallback entry before PC Teleport is ready) ----
    std::mt19937 rng(42);
    start_node_ = rng() % npts_;
    std::cout << "  Fallback start node: " << start_node_ << std::endl;

    // ---- Random insertion order (same permutation for both passes) ----
    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    // ================================================================
    // PRE-PASS: PC Teleport initialisation
    // ================================================================
    std::cout << "\n[PC Teleport] Computing first principal component..." << std::endl;
    Timer pca_timer;
    compute_pca(/*iters=*/20);
    std::cout << "  PCA done in " << pca_timer.elapsed_seconds()
              << " s.  Entry points are now query-adaptive." << std::endl;

    // ================================================================
    // PASS 1: alpha = 1.0  (local short-range graph)
    // ================================================================
    std::cout << "\n[Pass 1] Building local graph"
              << " (R=" << R << ", L=" << L << ", alpha=1.0, gamma=" << gamma
              << ")..." << std::endl;

    Timer pass1_timer;
    build_pass(perm, R, L, /*pass_alpha=*/1.0f, gamma, /*clear_graph=*/true);
    std::cout << "  Pass 1 complete in " << pass1_timer.elapsed_seconds()
              << " s." << std::endl;

    // ================================================================
    // PASS 2: alpha = user value  (long-range edges)
    // ================================================================
    std::cout << "\n[Pass 2] Refining with long-range edges"
              << " (R=" << R << ", L=" << L << ", alpha=" << alpha
              << ", gamma=" << gamma << ")..." << std::endl;

    Timer pass2_timer;
    // clear_graph=false: pass 2 starts from the pass-1 graph.
    build_pass(perm, R, L, alpha, gamma, /*clear_graph=*/false);
    std::cout << "  Pass 2 complete in " << pass2_timer.elapsed_seconds()
              << " s." << std::endl;

    // ---- Summary stats ----
    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; ++i) total_edges += graph_[i].size();
    double avg_degree = static_cast<double>(total_edges) / npts_;

    std::cout << "\n  Build complete." << std::endl;
    std::cout << "  Average out-degree: " << avg_degree << std::endl;
}

// ============================================================================
// Search  —  with PC Teleport entry point
// ============================================================================

SearchResult VamanaIndex::search(const float* query, uint32_t K,
                                 uint32_t L) const {
    if (L < K) L = K;

    // Use PC Teleport if PCA has been computed, otherwise fall back to the
    // fixed start node (e.g. after load() without recomputing PCA).
    uint32_t entry = projections_.empty()
                         ? start_node_
                         : get_entry_point(query);

    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L, entry);
    double latency = t.elapsed_us();

    SearchResult result;
    result.dist_cmps  = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); ++i)
        result.ids.push_back(candidates[i].second);

    return result;
}

// ============================================================================
// Save / Load
// ============================================================================
// Binary format:
//
//   Header (all uint32 unless noted):
//     npts | dim | start_node
//   Graph:
//     For each node i:  degree  |  neighbor_ids[degree]
//   PCA block (appended; load detects EOF gracefully if absent):
//     [uint32] dim             — sanity check
//     [float * dim] mean_vec
//     [float * dim] pc_vec
//     [uint32] npts            — number of projection entries
//     [float * npts] proj_vals — projection scalars, sorted
//     [uint32 * npts] proj_ids — corresponding point ids, sorted

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    // Header
    out.write(reinterpret_cast<const char*>(&npts_),       4);
    out.write(reinterpret_cast<const char*>(&dim_),        4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    // Graph
    for (uint32_t i = 0; i < npts_; ++i) {
        uint32_t deg = static_cast<uint32_t>(graph_[i].size());
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0)
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
    }

    // PCA block
    if (!pc_vec_.empty()) {
        uint32_t d = dim_;
        out.write(reinterpret_cast<const char*>(&d), 4);
        out.write(reinterpret_cast<const char*>(mean_vec_.data()),
                  dim_ * sizeof(float));
        out.write(reinterpret_cast<const char*>(pc_vec_.data()),
                  dim_ * sizeof(float));

        uint32_t n = static_cast<uint32_t>(projections_.size());
        out.write(reinterpret_cast<const char*>(&n), 4);
        for (const auto& [val, id] : projections_) {
            out.write(reinterpret_cast<const char*>(&val), 4);
            out.write(reinterpret_cast<const char*>(&id),  4);
        }
    }

    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    // Load data vectors
    FloatMatrix mat = load_fbin(data_path);
    npts_      = mat.npts;
    dim_       = mat.dims;
    data_      = mat.data.release();
    owns_data_ = true;

    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts),    4);
    in.read(reinterpret_cast<char*>(&file_dim),     4);
    in.read(reinterpret_cast<char*>(&start_node_),  4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index has " + std::to_string(file_npts) +
            "x" + std::to_string(file_dim) + ", data has " +
            std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    for (uint32_t i = 0; i < npts_; ++i) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0)
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
    }

    // Try to read optional PCA block
    {
        uint32_t d = 0;
        in.read(reinterpret_cast<char*>(&d), 4);
        if (in && d == dim_) {
            mean_vec_.resize(dim_);
            pc_vec_.resize(dim_);
            in.read(reinterpret_cast<char*>(mean_vec_.data()),
                    dim_ * sizeof(float));
            in.read(reinterpret_cast<char*>(pc_vec_.data()),
                    dim_ * sizeof(float));

            uint32_t n = 0;
            in.read(reinterpret_cast<char*>(&n), 4);
            projections_.resize(n);
            for (auto& [val, id] : projections_) {
                in.read(reinterpret_cast<char*>(&val), 4);
                in.read(reinterpret_cast<char*>(&id),  4);
            }
            std::cout << "  PC Teleport state loaded." << std::endl;
        }
    }

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_ << std::endl;
}