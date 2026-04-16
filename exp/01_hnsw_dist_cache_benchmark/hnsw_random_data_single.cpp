/*
    This code calls the function recall_rand_k(...)
    ONCE to generate a new set of random data for a test each time it's called.

    -DDISTCOUNT run with this macro to replace time with dist func calls to measure runtime, MUST USE if want to cache distance
    -DDISTCACHE run with this macro to cache distance between points

    WORKS DON'T TOUCH
*/

#include <fstream>
#include <chrono>

#include "../../hnswlib/hnswlib/hnswlib.h"
#include "../helpers.h"
#include "../helpers_hnsw.h"

/* CHANGE THIS PART EVERY TIME YOU WANT NEW LOG */
#define logfile "../../logresult/20251001-hnsw-dist-cache-test.log"
#define resultfile "../../logresult/20251001-hnsw-dist-cache-test.txt"

// new distance function wrapper with distance cache and function call counter
std::map<std::pair<const void*, const void*>, std::pair<float, size_t>> dist_cache_;
hnswlib::DISTFUNC<float> fstdistfunc_backup_ = nullptr;
size_t fstdistfunc_counter_ = 0;

float
fstdistfunc_cache_(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {

    std::pair<const void *, const void *> key = pVect1v < pVect2v 
                                                ? std::make_pair(pVect1v, pVect2v) 
                                                : std::make_pair(pVect2v, pVect1v);
    #ifdef DISTCACHE
        auto it = dist_cache_.find(key); // try to find key
        
        if (it != dist_cache_.end()) { // if found key in cache
            ++it->second.second;
            return it->second.first;
        }
        
        float distance = fstdistfunc_backup_(pVect1v, pVect2v, qty_ptr); // if not
        fstdistfunc_counter_++;
        dist_cache_[key] = {distance, 1};
        return distance;
    #else 
        // every time you call dist func it calculates distance anew
        // just increment counter
        float distance = fstdistfunc_backup_(pVect1v, pVect2v, qty_ptr);
        fstdistfunc_counter_++;
        dist_cache_[key].second++;
        return distance;
    #endif
}

// this returns recall for kNN
// find kNN for each and average out percentage
float recall_rand_k_cache(int testnum, int dim, int max_elements, int m, int ef_construction, int k, int numtest, 
                    std::ofstream& log, std::ofstream& result, hnswlib::DISTFUNC<float> newfunction) {

    // Initing index | EDIT : alg_brute to find ground-truth kNN
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, m, ef_construction);
    hnswlib::BruteforceSearch<float>* alg_brute = new hnswlib::BruteforceSearch<float>(&space, max_elements);

    fstdistfunc_backup_ = alg_hnsw->fstdistfunc_;

    #ifdef DISTCOUNT
        // CHANGE DIST FUNCTION
        alg_hnsw->replace_fstdistfunc_for_cache(newfunction);
        #ifdef DEBUG 
            std::cout << "function replaced" << std::endl;
        #endif
        result << "d,efc,m,efs,k,recall,distcountconstr,distcountsearch";
        #ifdef DISTCACHE
            result << ",distcache";
        #endif
        result << std::endl;
    #else
        auto starttime = std::chrono::high_resolution_clock::now();
        result << "d,efc,m,efs,k,recall,searchtime" << std::endl;
    #endif

    log << "recall_efs_k Test " << testnum << std::endl;
    log << "\tmax_elements = " << max_elements
        << "\tdim = " << dim
        << "\tef_construction = " << ef_construction 
        << "\tM = " << m
        << "\tef_search = " << k
        << "\tk = " << k << std::endl;

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    #ifdef DEBUG 
        std::cout << "data generated" << std::endl; 
    #endif

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        float* point_data = data + i * dim;
        alg_hnsw->addPoint(point_data, i);
        alg_brute->addPoint(point_data, i);
    }

    #ifdef DEBUG 
        std::cout << "data added to graph" << std::endl; 
    #endif

    #ifdef DISTCOUNT
        // get number of distfunc calls for construction
        size_t constr_count = fstdistfunc_counter_;
        // reset var
        fstdistfunc_counter_ = 0;
    #else
        auto endtime = std::chrono::high_resolution_clock::now();
        auto duration = (endtime - starttime).count(); 
        log << "\tConstruction time\t: " << duration << "ms" << std::endl;
        starttime = std::chrono::high_resolution_clock::now();
    #endif

    // Query the elements for themselves and measure recall
    // try to find the first numtest items
    float correct = 0;
    for (int i = 0; i < numtest; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> appro_result = alg_hnsw->searchKnn(data + i * dim, k);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> exact_result = alg_brute->searchKnn(data + i * dim, k);

        // EDIT : for each item in brute, find it in appro
        while(!exact_result.empty()) {
            if(foundinapprox(exact_result.top(), appro_result))
                correct++;
            exact_result.pop();
        }
    }

    #ifdef DEBUG 
        std::cout << "search complete" << std::endl; 
    #endif

    float recall = correct / (numtest * k);
    std::cout << "kNN Test " << testnum << ":\nRecall: " << recall << std::endl;

    #ifdef DISTCOUNT
        // get number of distfunc calls for construction
        size_t search_count = fstdistfunc_counter_;
        log << "\tDist func call constr\t: " << constr_count << std::endl;
        log << "\tDist func call search\t: " << search_count << std::endl;
        result << dim << "," << ef_construction << "," << m << "," << k << "," << k << "," << recall << "," << constr_count << "," << search_count << std::endl;
    #else
        endtime = std::chrono::high_resolution_clock::now();
        duration = (endtime - starttime).count(); 
        log << "\tSearch time\t\t\t: " << duration << "ms" << std::endl;
        log << "\tRecall\t\t\t\t: " << recall << std::endl << std::endl;
        result << dim << "," << ef_construction << "," << m << "," << k << "," << k << "," << recall << "," << duration << std::endl;     //"d,efc,m,efs,k,recall"
    #endif

    delete alg_hnsw;
    delete alg_brute;
    return 0;
}

int main() {
    // create log in append mode
    std::ofstream log(logfile, std::ios::app); 
    std::ofstream result(resultfile, std::ios::app); 
    
    int efs = 100, m = 128, efc = 200, d = 128; 
    recall_rand_k_cache(1, d, 1000000, m, efc, efs, 100, log, result, fstdistfunc_cache_);

    return 0;
}