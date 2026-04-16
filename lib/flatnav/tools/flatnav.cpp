/*
    This code runs hnsw benchmarking, measuring cost with number of distance calculations with distance cache.
    Only beam width changes between runs.

    To run :
    cd flatnav
    ./bin/build.sh -e -ins 100
    ./bin/build.sh -e -insm 10
    cd build
    ./flatnav
*/

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/index/Index.h>
#include "../../src/utils.h"
#include "../../src/datadef.h"
#include "../../src/outdef.h"
#include <iostream>
#include <chrono>
#include <random>
#include <cstdint>


/* -----EDIT THIS SECTION TO UPDATE FILE LOCATION & NUM-INIT----- */
// Define initializeSearch numbers
// const int SNUMINIT = 100;
#define SNINUM 1
int snuminit[] = {100};
const int CNUMINIT = 100;
/* -------------------------------------------------------------- */

#ifdef FILEAPP
    #define MODE app
#else
    #define MODE out
#endif

#if defined(INCM) && defined(INSM)
    #define RAND "-RandMsc"
#elif defined (INCM)
    #define RAND "-RandMc"
#elif defined (INSM)
    #define RAND "-RandMs"
#elif defined(INC) && INC == 1 && defined(INS) && INS == 1
    #define RAND "-Randsc"
#elif defined(INC) && INC == 1
    #define RAND "-Randc"
#elif defined(INS) && INS == 1
    #define RAND "-Rands"
#else
    #define RAND ""
#endif


void record_data(std::vector<float>& lat, std::vector<int>& distcomp,
        std::vector<int>& hopsmetric, std::vector<float>& recall, 
        std::ofstream& result, int dim, int efc, int m, int efs, int k, int qnum, int sni) {

    // Sort data and find p50, p99, ave for latency, distance computation and recall
    // Get
    std::sort(lat.begin(), lat.end());
    float lp50 = percentile<float>(lat, 50, qnum);
    float lp99 = percentile<float>(lat, 99, qnum);
    float lave = average<float>(lat);

    std::sort(distcomp.begin(), distcomp.end());
    float dp50 = percentile<int>(distcomp, 50, qnum);
    float dp99 = percentile<int>(distcomp, 99, qnum);
    float dave = average<int>(distcomp);

    std::sort(hopsmetric.begin(), hopsmetric.end()); // only 1 hop metric because there are no upper layers to traverse
    float hp50 = percentile<int>(hopsmetric, 50, qnum);
    float hp99 = percentile<int>(hopsmetric, 99, qnum);
    float have = average<int>(hopsmetric);

    std::sort(recall.begin(), recall.end());
    float rp50 = percentile<float>(recall, 50, qnum);
    float rp99 = percentile<float>(recall, 99, qnum);
    float rave = average<float>(recall);

    // Record
    std::cout << "Recall: " << rave << std::endl;

    // Save results
    result << dim << "," << efc << "," << m << "," << efs << "," << k << "," << sni << ","
            << lp50 << "," << lp99 << "," << lave << "," 
            << dp50 << "," << dp99 << "," << dave << "," 
            << hp50 << "," << hp99 << "," << have << "," 
            << rp50 << "," << rp99 << "," << rave
            << std::endl;
}


void run_flatnav_l2(int dim, float *data, float *qs, int *gt, int datasize, int qnum, int gtnum,
    int m, int k, int efc, int cnuminit, std::ofstream& result, std::string space, int dataind) {

    std::cout << "Creating L2 space" << std::endl;

    // Define distance
    auto distance = flatnav::distances::SquaredL2Distance<>::create(dim);

    // Build Index
    auto index = new flatnav::Index<flatnav::distances::SquaredL2Distance<>, int>(
                    /* dist = */ std::move(distance), 
                    /* dataset_size = */ datasize,
                    /* max_edges_per_node = */ m,
                    /* collect_stats */ true);

    // Set how many entry points to have, defined as a MACRO
    #ifdef INCM
        index->setNumEntryConstr(INCM);
    #endif
    #ifdef INSM
        index->setNumEntrySearch(INSM);
    #endif

    index->setNumThreads(NUMTHREAD);
    std::vector<int> labels(datasize);
    std::iota(labels.begin(), labels.end(), 0);

    // Add data to index
    index->addBatch<float>(/* data = */ (void *)data,
                            /* labels = */ labels,
                            /* ef_construction */ efc,
                            /*num_initializations = 100*/ cnuminit);

    // Set up performance measurement variables
    std::vector<float> lat(qnum);
    std::vector<int> distcomp(qnum);
    std::vector<int> hopsmetric(qnum);
    std::vector<float> recall(qnum);

    // Query setup
    float correct = 0.0;
    int testnum = 1, numrun = 2;
    
    // Run queries
    // For each numinit
    for (int n = 0; n < SNINUM; n++) {
        // For each efs
        for (int efsind = 0; efsind < efsnum; efsind++) {
            std::cout << "Running test " << testnum++ << std::endl;
            // std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;

            for (int i = 0; i < qnum; i++) {            // for each query item
                // Clean up stats
                index->resetStats(); // <<<<<<<<<<<<<<<<<<<<<<< REVISE FUNCTION
                correct = 0;
                float *q = qs + dim * i;                // q = this item
                int *g = gt + gtnum * i;                // g = ground truth set for this item. *(gt + i * K + j)
                                                        // need to separate k and num_gt because num_gt is set in incoming dataset gt, but k can vary
                // Run query
                std::vector<std::pair<float, int>> result;
                const std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();
                for (int j = 0; j < numrun; j++) 
                    result = index->search(q, k, efsarr[efsind],/*num_initializations = 100*/ snuminit[n]);
                const std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();

                // For each item in appro, find it in gt
                for (int j = 0; j < k; j++) {             
                    for (int l = 0; l < gtnum; l++) {            
                        if (result[j].second == g[l]) { // count correct neighbors
                            correct++;
                            break;
                        }
                    }
                }

                double numrund = double(numrun);
                // Latency
                double latency = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
                lat[i] = latency / numrund; //

                // Dist Comp
                distcomp[i] = index->distanceComputations() / numrund;
                
                // Hops
                hopsmetric[i] = index->metricHops() / numrund;
                
                // Recall
                recall[i] = correct * 1.0f / k;
            } // for (int i = 0; i < qnum; i++)

            // save  
            record_data(lat, distcomp, hopsmetric, recall, result, dim, efc, m, efsarr[efsind], k, qnum, snuminit[n]);
        } // for (int efsind = 0; efsind < efsnum; efsind++)
    } // for (int n = 0; n < SNINUM; n++)
    delete index;

}


// Function to build index and compute recall
void run_flatnav_ip(int dim, float *data, float *qs, int *gt, int datasize, int qnum, int gtnum,
    int m, int k, int efc, int cnuminit, std::ofstream& result, std::string space, int dataind) {
    
    std::cout << "Creating IP space" << std::endl;

    // Define distance
    auto distance = flatnav::distances::InnerProductDistance<>::create(dim);


    // Build Index
    auto index = new flatnav::Index<flatnav::distances::InnerProductDistance<>, int>(
                    /* dist = */ std::move(distance), 
                    /* dataset_size = */ datasize,
                    /* max_edges_per_node = */ m,
                    /* collect_stats */ true);

    // Set how many entry points to have, defined as a MACRO
    #ifdef INCM
        index->setNumEntryConstr(INCM);
    #endif
    #ifdef INSM
        index->setNumEntrySearch(INSM);
    #endif

    index->setNumThreads(NUMTHREAD);
    std::vector<int> labels(datasize);
    std::iota(labels.begin(), labels.end(), 0);

    // Add data to index
    index->addBatch<float>(/* data = */ (void *)data,
                            /* labels = */ labels,
                            /* ef_construction */ efc,
                            /*num_initializations = 100*/ cnuminit);

    // Set up performance measurement variables
    std::vector<float> lat(qnum);
    std::vector<int> distcomp(qnum);
    std::vector<int> hopsmetric(qnum);
    std::vector<float> recall(qnum);

    // Query setup
    float correct = 0.0;
    int testnum = 1, numrun = 2;
    
    // Run queries
    // For each numinit
    for (int n = 0; n < SNINUM; n++) {
        // For each efs
        for (int efsind = 0; efsind < efsnum; efsind++) {
            std::cout << "Running test " << testnum++ << std::endl;
            // std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
            // for each query item
            for (int i = 0; i < qnum; i++) {            
                // Clean up stats
                index->resetStats();
                correct = 0;
                float *q = qs + dim * i;                  // q = this item
                int *g = gt + gtnum * i;                 // g = ground truth set for this item. *(gt + i * K + j)
                                                // need to separate k and num_gt because num_gt is set in incoming dataset gt, but k can vary
                // Run query
                std::vector<std::pair<float, int>> result;
                const std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();
                for (int j = 0; j < numrun; j++) 
                    result = index->search(q, k, efsarr[efsind],/*num_initializations = 100*/ snuminit[n]);
                const std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();

                // For each item in appro, find it in gt
                for (int j = 0; j < k; j++) {             
                    for (int l = 0; l < gtnum; l++) {            
                        // std::cout << "result[" << j << "].second : " << result[j].second << "\tg[" << l << "] : " << g[l] << std::endl;
                        if (result[j].second == g[l]) {
                            correct++;
                            break;
                        }
                    }
                }

                double numrund = double(numrun);
                // Latency
                double latency = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
                lat[i] = latency / numrund; //

                // Dist Comp
                distcomp[i] = index->distanceComputations() / numrund;
                
                // Hops
                hopsmetric[i] = index->metricHops() / numrund;
                
                // Recall
                recall[i] = correct * 1.0f / k;
            }

            // save  
            record_data(lat, distcomp, hopsmetric, recall, result, dim, efc, m, efsarr[efsind], k, qnum, snuminit[n]);
        }
    }
    delete index;
    
}


int main(int argc, char **argv) {

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
        std::string resultfile = FNOUTPATH +
                                dataname[i][NAME] + ".txt"; // + RAND + "-s" + std::to_string(SNUMINIT) + "c" + std::to_string(CNUMINIT)
        std::ofstream result(resultfile, std::ios::MODE);
        std::cout << "creating result file" << std::endl;
        if(!result.is_open())
            throw std::runtime_error("main : result file not created");

        #ifndef FILEAPP
            result  << "d,efc,m,efs,k,sni,"
                    << "lp50,lp99,lave,"
                    << "dp50,dp99,dave," 
                    << "hbp50,hbp99,hbave," 
                    << "rp50,rp99,rave"
                    << std::endl;
        #endif

        // Run tests
        std::cout << "Start benchmarking for " << dataname[i][NAME] << std::endl;
        int m = metadata[i][M], efc = metadata[i][EFC];
        if(dataname[i][SPACE] == "L2")
            run_flatnav_l2(metadata[i][DIM], 
                        data, 
                        qs, 
                        gt, 
                        metadata[i][DATASIZE], 
                        metadata[i][QNUM], 
                        metadata[i][GTNUM], 
                        m,  
                        metadata[i][GTNUM], 
                        efc, 
                        CNUMINIT, 
                        result, 
                        dataname[i][SPACE],
                        i);
        else    // void run_flatnav_ip(int dim, float *data, float *qs, int *gt, int datasize, int qnum, int gtnum,
                // int m, int efc, int k, int snuminit, int cnuminit, std::ofstream& result, std::string space)
            run_flatnav_ip(metadata[i][DIM], // dim
                        data,                   // data
                        qs,                     // qs
                        gt,                     // gt
                        metadata[i][DATASIZE],  // datasize
                        metadata[i][QNUM],      // qnum
                        metadata[i][GTNUM],     // gtnum 
                        m,                      // m
                        metadata[i][GTNUM],     // k
                        efc,
                        CNUMINIT, 
                        result, 
                        dataname[i][SPACE],
                        i);    // space string
        
        // Save results
        result.close();

        // Clean up
        delete[] data;
        delete[] qs;
        delete[] gt;
    }

    return 0;
}