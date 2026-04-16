/*
    This code runs hnsw benchmarking, measuring cost with number of 
    distance calculations with distance cache.
    Only beam width changes between runs.
    Currently assumed to not be run with random initialization points.

    Edit src/datadef.h to select which dataset(s) to benchmark.

    To run :
    g++ hnsw.cpp -o hnsw
    ./hnsw

    This week's changes include :
        download deep data again 
        convert data by running 
            python convert_ann_benchmark_datasets.py data/data.h5py --normalize

        first run normally without optimization matching
        then run with optimization matching
        run for both sift and deep, con&search parameters matched
*/

#include <fstream>
#include <chrono>

#include "../../hnswlib/hnswlib/hnswalg.h"
#include "../../src/utils.h"
#include "../../src/datadef.h"
#include "../../src/outdef.h"
#include "../helpers_hnsw.h"

#define RAND ""

#ifdef FILEAPP
    #define MODE app
#else
    #define MODE out
#endif

#ifdef OPT
    #define OPTFLAG "-opt"
#else
    #define OPTFLAG ""
#endif


// global variables to store cache
hnswlib::DISTFUNC<float> fstdistfunc_backup_ = nullptr;
size_t fstdistfunc_counter = 0;

// new distance function wrapper with distance cache and function call counter
float fstdistfunc_cache_(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float distance = fstdistfunc_backup_(pVect1v, pVect2v, qty_ptr);
    fstdistfunc_counter++;
    return distance;
}

// Function returns recall for kNN search as average recall percentage
float run_hnsw(int dim, int max_elements, int k, int qnum, int m, int efc,
                    std::ofstream& result, hnswlib::DISTFUNC<float> newfunction,
                    float *data, float *qs, int *gt, std::string space, int i) {

    // Define distance 
    hnswlib::SpaceInterface<float> *distance = nullptr;
    if(space == "L2") {
        std::cout << "\tCreating L2 space..." << std::endl;
        distance = new hnswlib::L2Space(dim);
    }
    else {
        std::cout << "\tCreating IP space..." << std::endl;
        distance = new hnswlib::InnerProductSpace(dim);
    }
    
    // Build index
    std::cout << "\tInitializing index..." << std::endl;
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(distance, max_elements, m, efc);

    // Performance measurements : swap out original distance function to count distance computation
    fstdistfunc_backup_ = alg_hnsw->fstdistfunc_;
    alg_hnsw->replace_fstdistfunc_for_cache(newfunction);
    #ifdef DEBUG 
        std::cout << "function replaced" << std::endl;
    #endif

    // Add data to index
    std::cout << "\tAdding points..." << std::endl;
    for (int i = 0; i < max_elements; i++) {
        float* point_data = data + i * dim;
        alg_hnsw->addPoint(point_data, i);
    }

    // Set up performance measurement variables
    fstdistfunc_counter = 0;
    std::vector<float> lat(qnum);
    std::vector<int> distcomp(qnum);
    std::vector<int> hops(qnum);
    std::vector<float> recall(qnum);

    // Query setup
    std::cout << "\tStarting benchmarking for " << dataname[i][NAME] << "...\n";
    float correct = 0, val = 0;
    int testnum = 1, numrun = 2; /* number of times to run queries to average out unexpected performance disturbances */

    // For each efs
    for (int efsind=0; efsind < EFSNUM; efsind++) {
        std::cout << "\tRunning test " << testnum << "...\n";

        alg_hnsw->setEf(efsarr[efsind]);

        // For each query point
        for (int i = 0; i < qnum; i++) {
            // Clean up stats
            alg_hnsw->resetMetrics();
            correct = 0;
            fstdistfunc_counter = 0;

            // Run query
            const std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();
            std::priority_queue<std::pair<float, hnswlib::labeltype>> appro_result;
            for (int r = 0; r < numrun; r++)
                appro_result = alg_hnsw->searchKnn(qs + i * dim, k);
            const std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();

            // For each item in appro, find it in gt
            while(!appro_result.empty())        
            { 
                val = appro_result.top().second;
                for (int j = 0; j < k; j++)
                    if (val == gt[i * k + j]) {
                        correct++;
                        break;
                    }
                appro_result.pop();
            }
            
            double numrund = double(numrun);
            // Latency
            double latency = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
            lat[i] = latency / numrund; //

            // Dist Comp
            distcomp[i] = alg_hnsw->metric_distance_computations / numrund; //

            // Hops
            hops[i] = alg_hnsw->metric_hops / numrund;

            // Recall
            recall[i] = correct * 1.0f / k; 
        }

        // Get
        std::cout << "\t\t\tComputing results and saving...\n";
        std::sort(lat.begin(), lat.end());
        float lp50 = percentile<float>(lat, 50, qnum);
        float lp99 = percentile<float>(lat, 99, qnum);
        float lave = average<float>(lat);

        std::sort(distcomp.begin(), distcomp.end());
        float dp50 = percentile<int>(distcomp, 50, qnum);
        float dp99 = percentile<int>(distcomp, 99, qnum);
        float dave = average<int>(distcomp);

        std::sort(hops.begin(), hops.end());
        float hp50 = percentile<int>(hops, 50, qnum);
        float hp99 = percentile<int>(hops, 99, qnum);
        float have = average<int>(hops);

        std::sort(recall.begin(), recall.end());
        float rp50 = percentile<float>(recall, 50, qnum);
        float rp99 = percentile<float>(recall, 99, qnum);
        float rave = average<float>(recall);

        // Record
        std::cout << "\t\t\tRecall: " << rave << std::endl;

        result << dim << "," << efc << "," << m << "," << efsarr[efsind] << "," << k << "," 
                << lp50 << "," << lp99 << "," << lave << "," 
                << dp50 << "," << dp99 << "," << dave << "," 
                << hp50 << "," << hp99 << "," << have << "," 
                << rp50 << "," << rp99 << "," << rave
                << std::endl;
    
        testnum++;
    }

    delete alg_hnsw; 
    delete distance;
    return 0;
}


int main() {
    // Array variables
    float *data = nullptr, *qs = nullptr;
    int *gt = nullptr;

    // For each dataset
    for (int i = 0; i < DATASETS; i++)
    {
        std::cout << "Benchmarking " << dataname[i][NAME] << "...\n";

        // Try block for reading xvec file to arr
        std::cout << "\tLoading data..." << std::endl;
        try {
        xvecfile_to_arr<float>(DATAPATH, dataname[i][NAME], "data.fvecs", &data, metadata[i][DATASIZE], metadata[i][DIM]);
        xvecfile_to_arr<float>(DATAPATH, dataname[i][NAME], "qs.fvecs", &qs, metadata[i][QNUM], metadata[i][DIM]);
        xvecfile_to_arr<int>(DATAPATH, dataname[i][NAME], "gt.ivecs", &gt, metadata[i][QNUM], metadata[i][GTNUM]);

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

        // Create result file
        std::string resultfile = std::string(HNSWOUTPATH) +
                                dataname[i][NAME] + OPTFLAG + ".txt";
        std::ofstream result(resultfile, std::ios::MODE);

        #ifndef FILEAPP
            result  << "d,efc,m,efs,k,"
                    << "lp50,lp99,lave, "
                    << "dp50,dp99,dave," 
                    << "hp50,hp99,have," 
                    << "rp50,rp99,rave"
                    << std::endl;
        #endif

        // Run tests with increasing beam width
        int m = metadata[i][M], efc =  metadata[i][EFC];
        run_hnsw(metadata[i][DIM],          // dim
                metadata[i][DATASIZE],      // max_elements
                metadata[i][GTNUM],         // k
                metadata[i][QNUM],          // qnum
                m, efc, 
                result,                     // output file
                fstdistfunc_cache_,         // distance function
                data, qs, gt,               // input arrays
                dataname[i][SPACE],         // dataset space
                i);
        
        // Save results
        result.close();

        // Clean up
        delete[] data;
        delete[] qs;
        delete[] gt;
    }
    return 0;
}