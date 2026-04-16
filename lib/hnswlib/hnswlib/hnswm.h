#include "hnswalg.h"
#include "hnswlib.h"

namespace hnswlib {

template<typename dist_t>
class hnswM {
    // private
    // public
    public:
    /* VARS */
    // graphs
        HierarchicalNSW<dist_t>* hnsw1;
        HierarchicalNSW<dist_t>* hnsw2;
    // cache
        std::map<tableint/*candidate_id*/, dist_t/*distance*/>* dist_cache_;

    /* CONSTRUCTORS */
    hnswM<dist_t> () {
        hnsw1 = nullptr;
        hnsw2 = nullptr;
        dist_cache_ = new std::map<tableint, dist_t>();
    }
    hnswM<dist_t> (
        SpaceInterface<dist_t> *s,
        size_t max_elements,

        size_t M1 = 16,
        size_t ef_construction1 = 200,
        size_t random_seed1 = 100,

        size_t M2 = 16,
        size_t ef_construction2 = 200,
        size_t random_seed2 = 100
    ) {
        hnsw1 = new HierarchicalNSW<dist_t> (
            s, max_elements, M1, ef_construction1, random_seed1, dist_cache_
        );
        hnsw2 = new HierarchicalNSW<dist_t> (
            s, max_elements, M2, ef_construction2, random_seed2, dist_cache_
        );
        dist_cache_ = new std::map<tableint, dist_t>();
    }

    // reset
    void resetHnswSet() {
        delete hnsw1;
        delete hnsw2;
        hnsw1 = nullptr;
        hnsw2 = nullptr;
        dist_cache_->clear();
    }
    
    // destroyer
    ~hnswM() {
        delete hnsw1;
        delete hnsw2;
        delete dist_cache_;
    }

    // cache functions
    void insertCache(tableint id, dist_t dist) {
        (*dist_cache_)->insert({id, dist});
    }
    void updateCache(tableint id, dist_t dist) {
        (*dist_cache_)[id] = dist;
    }
    void resetCache() {
        dist_cache_->clear();
    }
    std::map<tableint, dist_t>* getCache() {
        return dist_cache_;
    }
};

}