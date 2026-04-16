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
        - type error corrected
*/

#include <fstream>
#include <chrono>

#include "../../hnswlib/hnswlib/hnswlib.h"
#include "../../src/utils.h"
#include "../../src/datadef.h"
#include "../../src/outdef.h"
#include "../helpers_hnsw.h"

// Define results file name variables

// #if defined(INC) && INC == 1 && defined(INS) && INS == 1
//   #define RAND "-scRand"
// #elif defined(INC) && INC == 1
//   #define RAND "-cRand"
// #elif defined(INS) && INS == 1
//   #define RAND "-sRand"
// #else
//   #define RAND ""
// #endif
#define RAND ""


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
                    float *data, float *qs, int *gt, std::string space, int dataind) {

    // Define distance 
    hnswlib::SpaceInterface<float> *distance = nullptr;
    if(space == "L2") {
        std::cout << "Creating L2 space" << std::endl;
        distance = new hnswlib::L2Space(dim);
    }
    else {
        std::cout << "Creating IP space" << std::endl;
        distance = new hnswlib::InnerProductSpace(dim);
    }
    
    // Build index
    std::cout << "Initializing index" << std::endl;
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(distance, max_elements, m, efc);

    // Performance measurements : swap out original distance function to count distance computation
    fstdistfunc_backup_ = alg_hnsw->fstdistfunc_;
    alg_hnsw->replace_fstdistfunc_for_cache(newfunction);
    #ifdef DEBUG 
        std::cout << "function replaced" << std::endl;
    #endif

    // Add data to index
    std::cout << "Adding points" << std::endl;
    for (int i = 0; i < max_elements; i++) {
        float* point_data = data + i * dim;
        alg_hnsw->addPoint(point_data, i);
    }

    // Set up performance measurement variables
    fstdistfunc_counter = 0;
    std::vector<float> lat(qnum);
    std::vector<float> latupper(qnum);
    std::vector<float> latbase(qnum);
    std::vector<int> distcomp(qnum);
    std::vector<int> distcompmetric(qnum);
    std::vector<int> distcompupper(qnum);
    std::vector<int> distcompbase(qnum);
    std::vector<int> hopsmetric(qnum);
    std::vector<int> hopsupper(qnum);
    std::vector<int> hopsbase(qnum);
    std::vector<float> recall(qnum);

    // Query setup
    float correct = 0, val = 0;
    int testnum = 1, numrun = 2; /* number of times to run queries to average out unexpected performance disturbances */

    // For each efs
    for (int efs=metadata[dataind][EFSLOW]; efs<=metadata[dataind][EFSHIGH]; efs+=metadata[dataind][STEP]) {
        std::cout << "Running test " << testnum++ << std::endl;

        alg_hnsw->setEf(efs);

        // For each query point
        for (int i = 0; i < qnum; i++) {
            // Clean up stats
            alg_hnsw->resetMetrics();
            correct = 0;
            fstdistfunc_counter = 0;

            // Run query
            const std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();
            std::priority_queue<std::pair<float, hnswlib::labeltype>> appro_result;
            for (int j = 0; j < numrun; j++)
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
            latupper[i] = alg_hnsw->metric_latency_upper / numrund; //
            latbase[i] = alg_hnsw->metric_latency_base / numrund; //

            // Dist Comp
            distcomp[i] = fstdistfunc_counter / numrund; //
            distcompupper[i] = alg_hnsw->metric_distcomp_upper / numrund; //
            distcompbase[i] = alg_hnsw->metric_distcomp_base / numrund; //
            distcompmetric[i] = alg_hnsw->metric_distance_computations / numrund; //

            // Hops
            hopsmetric[i] = alg_hnsw->metric_hops / numrund;
            hopsupper[i] = alg_hnsw->metric_hops_upper / numrund;
            hopsbase[i] =  alg_hnsw->metric_hops_base / numrund;

            // Recall
            recall[i] = correct * 1.0f / k; //
        }

        // Get
        std::sort(lat.begin(), lat.end());
        float lp50 = percentile<float>(lat, 50, qnum);
        float lp99 = percentile<float>(lat, 99, qnum);
        float lave = average<float>(lat);

        std::sort(latupper.begin(), latupper.end());
        float lup50 = percentile<float>(latupper, 50, qnum);
        float lup99 = percentile<float>(latupper, 99, qnum);
        float luave = average<float>(latupper);

        std::sort(latbase.begin(), latbase.end());
        float lbp50 = percentile<float>(latbase, 50, qnum);
        float lbp99 = percentile<float>(latbase, 99, qnum);
        float lbave = average<float>(latbase);

        std::sort(distcomp.begin(), distcomp.end());
        float dp50 = percentile<int>(distcomp, 50, qnum);
        float dp99 = percentile<int>(distcomp, 99, qnum);
        float dave = average<int>(distcomp);

        std::sort(distcompupper.begin(), distcompupper.end());
        float dup50 = percentile<int>(distcompupper, 50, qnum);
        float dup99 = percentile<int>(distcompupper, 99, qnum);
        float duave = average<int>(distcompupper);

        std::sort(distcompbase.begin(), distcompbase.end());
        float dbp50 = percentile<int>(distcompbase, 50, qnum);
        float dbp99 = percentile<int>(distcompbase, 99, qnum);
        float dbave = average<int>(distcompbase);

        std::sort(distcompmetric.begin(), distcompmetric.end());
        float dmp50 = percentile<int>(distcompmetric, 50, qnum);
        float dmp99 = percentile<int>(distcompmetric, 99, qnum);
        float dmave = average<int>(distcompmetric);

        std::sort(hopsupper.begin(), hopsupper.end());
        float hup50 = percentile<int>(hopsupper, 50, qnum);
        float hup99 = percentile<int>(hopsupper, 99, qnum);
        float huave = average<int>(hopsupper);

        std::sort(hopsbase.begin(), hopsbase.end());
        float hbp50 = percentile<int>(hopsbase, 50, qnum);
        float hbp99 = percentile<int>(hopsbase, 99, qnum);
        float hbave = average<int>(hopsbase);

        std::sort(hopsmetric.begin(), hopsmetric.end());
        float hmp50 = percentile<int>(hopsmetric, 50, qnum);
        float hmp99 = percentile<int>(hopsmetric, 99, qnum);
        float hmave = average<int>(hopsmetric);

        std::sort(recall.begin(), recall.end());
        float rp50 = percentile<float>(recall, 50, qnum);
        float rp99 = percentile<float>(recall, 99, qnum);
        float rave = average<float>(recall);

        // Record
        std::cout << "kNN Test " << testnum << ":\nRecall: " << rave << std::endl;

        result << dim << "," << efc << "," << m << "," << efs << "," << k << "," 
                << lp50 << "," << lp99 << "," << lave << "," 
                << lup50 << "," << lup99 << "," << luave << "," 
                << lbp50 << "," << lbp99 << "," << lbave << "," 
                << dp50 << "," << dp99 << "," << dave << "," 
                << dup50 << "," << dup99 << "," << duave << "," 
                << dbp50 << "," << dbp99 << "," << dbave << "," 
                << dmp50 << "," << dmp99 << "," << dmave << "," 
                << hup50 << "," << hup99 << "," << huave << "," 
                << hbp50 << "," << hbp99 << "," << hbave << "," 
                << hmp50 << "," << hmp99 << "," << hmave << "," 
                << rp50 << "," << rp99 << "," << rave

                // << ((dave - dmave < 0.1) ? "T" : "F")
                // << ((dmave - duave - dbave < 0.1) ? "T" : "F")
                // << ((hmave - huave - hbave < 0.1) ? "T" : "F")
                // << ((lave > luave + lbave) ? "T" : "F")
                << std::endl;
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
        // Try block for reading xvec file to arr
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
        std::string resultfile = HNSWOUTPATH +
                                dataname[i][NAME] + // RAND + 
                                // "-s" + std::to_string(SNUMINIT) + "c" + std::to_string(CNUMINIT) + 
                                ".txt";
        std::ofstream result(resultfile, std::ios::out);
        result  << "d,efc,m,efs,k,"
                << "lp50,lp99,lave, "
                << "lup50,lup99,luave,"
                << "lbp50,lbp99,lbave,"
                << "dp50,dp99,dave," 
                << "dup50,dup99,duave," 
                << "dbp50,dbp99,dbave,"
                << "dmp50,dmp99,dmave,"
                << "hup50,hup99,huave," 
                << "hbp50,hbp99,hbave," 
                << "hmp50,hmp99,hmave," 
                << "rp50,rp99,rave"
                // << "(dave - dmave < 0.1)"
                // << "(dmave - duave - dbave < 0.1)"
                // << "(hmave - huave - hbave < 0.1)"
                // << "(lave > cust_lat_upper + cust_lat_base)"
                << std::endl;

        // Run tests with increasing beam width
        std::cout << "Start benchmarking for " << dataname[i][NAME] << std::endl;
        int m = metadata[i][M], efc =  metadata[i][EFC];
        run_hnsw(metadata[i][DIM], // dim
                metadata[i][DATASIZE], // max_elements
                metadata[i][GTNUM], // k
                metadata[i][QNUM], // qnum
                m, efc, 
                result, // output file
                fstdistfunc_cache_, // distance function
                data, qs, gt, // input arrays
                dataname[i][SPACE], // dataset space
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