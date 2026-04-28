#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <string>

// Result of a single query search.
struct SearchResult {
    std::vector<uint32_t> ids;  // nearest neighbor IDs (sorted by distance)
    uint32_t dist_cmps;         // number of distance computations
    double latency_us;          // search latency in microseconds
};

// Vamana graph-based approximate nearest neighbor index.
//
// Enhancements over the baseline:
//
//   1. TWO-PASS CONSTRUCTION
//      build() runs two insertion passes over the data.  Pass 1 uses alpha=1.0,
//      biasing the RNG rule toward short-range (local) edges.  Pass 2 reruns
//      insertion with the user alpha (>= 1.0), opening long-range edges that
//      improve navigability.  Pass 2 starts from the graph left by pass 1.
//
//   2. PC TELEPORT
//      Each query is teleported to the dataset point whose scalar projection
//      onto the first principal component (PC1) is nearest to the query's own
//      projection.  PC1 is computed once during build via power iteration.
//      This gives a much better initial position for greedy search, reducing
//      the number of hops needed to reach the true neighbours.
//
class VamanaIndex {
  public:
    VamanaIndex() = default;
    ~VamanaIndex();

    // ---- Build ----
    // Loads data from an fbin file and builds the Vamana graph using
    // two-pass construction and PC Teleport.
    void build(const std::string& data_path, uint32_t R, uint32_t L,
               float alpha, float gamma);

    // ---- Search ----
    // Search for K nearest neighbors of a query vector.
    // Uses PC Teleport to pick the entry point for each query.
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    // ---- Persistence ----
    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_; }

  private:
    // A candidate = (distance, node_id). Ordered by distance.
    using Candidate = std::pair<float, uint32_t>;

    // ---- Core algorithms ----

    // Greedy search starting from a caller-supplied entry node.
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L, uint32_t entry) const;

    // Standard alpha-RNG pruning.
    void robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                      float alpha, uint32_t R);

    // Single insertion pass. clear_graph=true resets all adjacency lists.
    void build_pass(const std::vector<uint32_t>& perm,
                    uint32_t R, uint32_t L,
                    float pass_alpha, float gamma,
                    bool clear_graph);

    // ---- PC Teleport helpers ----
    void compute_pca(int iters = 20);
    uint32_t get_entry_point(const float* query) const;

    // ---- Data ----
    float*   data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

    // ---- Graph ----
    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    // ---- Concurrency ----
    mutable std::vector<std::mutex> locks_;

    // ---- PC Teleport state ----
    std::vector<float> mean_vec_;
    std::vector<float> pc_vec_;
    std::vector<std::pair<float, uint32_t>> projections_;  // sorted by projection

    // ---- Helpers ----
    const float* get_vector(uint32_t id) const {
        return data_ + static_cast<size_t>(id) * dim_;
    }
    const float* row(uint32_t id) const { return get_vector(id); }
};