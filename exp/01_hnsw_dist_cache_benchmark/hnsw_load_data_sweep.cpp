/*
    This code calls the function recall_rand_k(...)
    ONCE to generate a new set of random data for a test each time it's called.

    -DDISTCOUNT run with this macro to replace time with dist func calls to measure runtime, MUST USE if want to cache distance
    -DDISTCACHE run with this macro to cache distance between points

    Things to edit :
    1. log file name    : cache / nocache taken care of here, but change dataset name each time
    2. write option     : out or app
    3. dataset info     : name, datasize, qs, gt sizes
    4. test running     : loop for multiple tests / single fast test
*/

#include <fstream>
#include <chrono>

#include "../../hnswlib/hnswlib/hnswlib.h"
#include "../helpers.h"
#include "../helpers_hnsw.h"

/* LOG FILE */
#ifdef DISTCACHE
    // #define logfile "../../logresult/20251002-hnsw-dist-cache-gen.log"
    #define resultfile "../../logresult/20251002-hnsw-dist-cache-sift.txt"
#else
    // #define logfile "../../logresult/20251002-hnsw-dist-nocache-gen.log"
    #define resultfile "../../logresult/20251002-hnsw-dist-nocache-sift.txt"
#endif

/* WRITE OPTION */
#define OPT out

/* DATASET INFO */
#define DATAFILENAME "../../data/sift1m/sift10k_base.bin"
#define QSFILENAME "../../data/sift1m/sift10k_query100.bin"
#define GTFILENAME "../../data/sift1m/sift10k_groundtruth100.bin"

#define DIM 128
#define DATASIZE 10000
#define QNUM 100
#define GTNUM 100

// global variables to store cache
std::map<std::pair<const void*, const void*>, std::pair<float, size_t>> dist_cache_;
hnswlib::DISTFUNC<float> fstdistfunc_backup_ = nullptr;
size_t fstdistfunc_counter_ = 0;

// new distance function wrapper with distance cache and function call counter
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

float recall_data_k_cache(int testnum, int dim, int max_elements, int k, int qnum, int m, int efc, int efs,
                    std::ofstream& result, hnswlib::DISTFUNC<float> newfunction,
                    float *data, float *qs, int *gt) {

    // Initing index | EDIT : alg_brute to find ground-truth kNN
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, m, efc);
    alg_hnsw->setEf(efs);

    fstdistfunc_backup_ = alg_hnsw->fstdistfunc_;

    #ifdef DISTCOUNT
        // CHANGE DIST FUNCTION
        alg_hnsw->replace_fstdistfunc_for_cache(newfunction);
        #ifdef DEBUG 
            std::cout << "function replaced" << std::endl;
        #endif
    #else
        auto starttime = std::chrono::high_resolution_clock::now();
    #endif

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        float* point_data = data + i * dim;
        alg_hnsw->addPoint(point_data, i);
    }

    #ifdef DISTCOUNT
        // get number of distfunc calls for construction
        size_t constr_count = fstdistfunc_counter_;
        // reset var
        fstdistfunc_counter_ = 0;
    #else
        auto endtime = std::chrono::high_resolution_clock::now();
        auto duration = (endtime - starttime).count(); 
        // log << "\tConstruction time\t: " << duration << "ms" << std::endl;
        starttime = std::chrono::high_resolution_clock::now();
    #endif

    // Query the elements for themselves and measure recall
    // try to find the first numtest items
    float correct = 0, val = 0;

    for (int i = 0; i < qnum; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> appro_result = alg_hnsw->searchKnn(qs + i * dim, k);

        while(!appro_result.empty())        // for each item in appro, find it in gt
        { 
            val = appro_result.top().second;
            for (int j = 0; j < k; j++)
            {
                if (val == gt[i * k + j])
                {
                    correct++;
                    break;
                }
            }
            appro_result.pop();
        }
    }

    float recall = correct / (qnum * k);
    std::cout << "kNN Test " << testnum << ":\nRecall: " << recall << std::endl;

    #ifdef DISTCOUNT
        // get number of distfunc calls for construction
        size_t search_count = fstdistfunc_counter_;
        // log << "\tDist func call constr\t: " << constr_count << std::endl;
        // log << "\tDist func call search\t: " << search_count << std::endl;
        // result << "d,efc,m,efs,k,recall,distcountconstr,distcountsearch";
        result << dim << "," << efc << "," << m << "," << k << "," << k << "," << recall << "," << constr_count << "," << search_count << std::endl;
    #else
        endtime = std::chrono::high_resolution_clock::now();
        duration = (endtime - starttime).count(); 
        // log << "\tSearch time\t\t\t: " << duration << "ms" << std::endl;
        // log << "\tRecall\t\t\t\t: " << recall << std::endl << std::endl;
        result << dim << "," << efc << "," << m << "," << efs << "," << k << "," << recall << "," << duration << std::endl;     //"d,efc,m,efs,k,recall"
    #endif

    delete alg_hnsw; 
    return 0;
}

int main() {
    // create log in append mode
    // std::ofstream log(logfile, std::ios::OPT);
    std::ofstream result(resultfile, std::ios::OPT);

    // load data
    std::ifstream datafile(DATAFILENAME, std::ios::binary);
    std::ifstream qsfile(QSFILENAME, std::ios::binary);
    std::ifstream gtfile(GTFILENAME, std::ios::binary);
    if(!datafile | !qsfile | !gtfile)
      throw std::runtime_error("Failed to open datafile in main.\n");

    // fvec/ivec/bin to arrays
    float *data=nullptr, *qs=nullptr;
    int *gt=nullptr;

    try { 
        data = bin_to_arr<float>(datafile, DATAFILENAME, DATASIZE);
        qs = bin_to_arr<float>(qsfile, QSFILENAME, QNUM);
        gt = bin_to_arr<int>(gtfile, GTFILENAME, QNUM);

    #ifdef DEBUG
        std::cout << "Data validation:" << std::endl;
        std::cout << "First data point: ";
        for (int i = 0; i < 5; i++) std::cout << data[i] << " ";
        std::cout << std::endl;

        std::cout << "First query point: ";
        for (int i = 0; i < 5; i++) std::cout << qs[i] << " ";
        std::cout << std::endl;

        std::cout << "First ground truth: ";
        for (int i = 0; i < 5; i++) std::cout << gt[i] << " ";
        std::cout << std::endl;
    #endif

        if (data == nullptr || qs == nullptr || gt == nullptr) {
            throw std::runtime_error("Pointers are still null after allocation");
        }
    } catch (const std::exception& e) {
        std::cout << "Error : " << e.what() << std::endl;

        delete[] data;
        delete[] qs;
        delete[] gt;
        return 1;
    };

    datafile.close();
    qsfile.close();
    gtfile.close();

    #ifdef DISTCOUNT
        // CHANGE DIST FUNCTION
        result << "d,efc,m,efs,k,recall,distcountconstr,distcountsearch";
        #ifdef DISTCACHE
            result << ",distcache";
        #endif
        result << std::endl;
    #else
        result << "d,efc,m,efs,k,recall,searchtime" << std::endl;
    #endif

    int d = 128, testnum = 1;
    for (int efs = 100; efs <= 400; efs*=2) // must be >= k *2
    {
        for (int m = 32; m <=128; m*=2)
        {
            for (int efc = 50; efc <= 200; efc *= 2)
            {
                recall_data_k_cache(testnum++, d, DATASIZE, GTNUM, QNUM, m, efc, efs, result, fstdistfunc_cache_, data, qs, gt);
                dist_cache_.clear();
                fstdistfunc_counter_ = 0;
            }
        }
    }
    // int m = 32, efc = 50, efs = 50, d = 128, datasize = DATASIZE;
    // recall_data_k_cache(1, d, DATASIZE, GTNUM, QNUM, m, efc, efs, result, fstdistfunc_cache_, data, qs, gt);
    // dist_cache_.clear();

    result.close();

    delete[] data;
    delete[] qs;
    delete[] gt;
    return 0;
}