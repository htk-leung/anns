/*
    This code is the first pass in running multiple hnsw graphs and combining results

    hnswSI implements 3 different search initializations:
    - regular
    - efs=1 greedy search on base layer from normal starting point followed by normal beam search
    - flatnav method
*/

#include <fstream>
#include <chrono>

#include "../../hnswlib/hnswlib/hnswalg.h"
#include "../../src/utils.h"
#include "../../src/datadef.h"
#include "../../src/outdef.h"
#include "../helpers_hnsw.h"


#ifdef FILEAPP 
    #define FILEMODE "a"
#else
    #define FILEMODE "w"
#endif

#ifdef OPT
    #define OPTFLAG "-opt"
#else
    #define OPTFLAG ""
#endif

#define MODELTAG "si-"


// void compute_recall(std::vector<float>& recall, 
//     std::vector<std::vector<hnswlib::labeltype>>& m, 
//     int *gt, int qnum, int k) 
// {
//     std::cout << "\t\t\tComputing recall...\n";

//     int s = 0, r;
//     // for each query
//     for (int i = 0; i < qnum; i++) { 
//         s = m[i].size();
//         r = 0;

//         // for each thing found
//         for (int j = 0; j < s; j++) {
//             // loop through gt to find match
//             for (int n = 0; n < k; n++) {
//                 if (m[i][j] == gt[i * k + n]) {
//                     r++;
//                     break;
//                 }
//             }
//         }
//         recall[i] = float(r) / k;
//     }        
// }


void save_results(
    std::priority_queue<std::pair<float, hnswlib::labeltype>>& result,
    std::vector<float>& lat,
    std::vector<int>& distcomp,
    std::vector<float>& recall,
    int& numrun, int& i,
    int& k, int *gt,
    const std::chrono::time_point<std::chrono::high_resolution_clock> t1,
    const std::chrono::time_point<std::chrono::high_resolution_clock> t2,
    hnswlib::HierarchicalNSW<float>* hnsw)
{
    // Latency
    double latency = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
    lat[i] = latency / double(numrun);

    // Dist Comp
    distcomp[i] = hnsw->metric_distance_computations;

    // Recall
    // for each thing found
    int r = 0;
    while(!result.empty()) {
        // loop through gt to find match
        auto label = result.top().second;
        for (int n = 0; n < k; n++) {
            if (label == gt[i * k + n]) {
                r++;
                break;
            }
        }
        result.pop();
    }
    recall[i] = r ? float(r) / k : 0;
}


void compute_and_save(
    int d, int efc, int m, int efs, int k,
    std::vector<float>& lat,
    std::vector<int>& distcomp,
    std::vector<float>& recall,
    int qnum, int testnum,
    FILE* result)
{
    std::cout << "\t\t\tComputing the rest and saving...\n";
    
    // Get
    std::sort(lat.begin(), lat.end());
    float lp50 = percentile<float>(lat, 50, qnum);
    float lp99 = percentile<float>(lat, 99, qnum);
    float lave = average<float>(lat);

    std::sort(distcomp.begin(), distcomp.end());
    float dp50 = percentile<int>(distcomp, 50, qnum);
    float dp99 = percentile<int>(distcomp, 99, qnum);
    float dave = average<int>(distcomp);

    std::sort(recall.begin(), recall.end());
    float rp50 = percentile<float>(recall, 50, qnum);
    float rp99 = percentile<float>(recall, 99, qnum);
    float rave = average<float>(recall);

    // Print
    std::cout << "\t\t\tRecall: " << rave << std::endl;

    // Save to result file
    std::fprintf(result, "%4d,%8d,%8d,%8d,%8d,%8.2f,%8.2f,%8.2f,%12.2f,%12.2f,%12.2f,%8.2f,%8.2f,%8.2f\n", 
        d,efc,m,efs,k,lp50,lp99,lave,dp50,dp99,dave,rp50,rp99,rave);
}


// Function returns recall for kNN search as average recall percentage
float run_hnsw(int dim, int efc, int m, int efs, int k, int qnum,
    float *qs, int *gt, 
    hnswlib::HierarchicalNSW<float>* hnsw,
    FILE **resultsFP, std::vector<std::string> test_tag) 
{
    // Query setup
    int numrun = 2; /* number of times to run queries to average out unexpected performance disturbances */

    // latency regular
    std::vector<float> lat(qnum);
    // distcomp regular
    std::vector<int> dc(qnum);
    // recall regular
    std::vector<float> recall(qnum, 0);

    // For each test 
    for (int tt = 0; tt < test_tag.size(); tt++) {

        // For each query point
        for (int i = 0; i < qnum; i++) {
            // Results priority queue
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result;

            const std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();

            // Run twice
            for (int n = 0; n < numrun; n++) {
                // Clear metrics
                hnsw->resetMetrics();
                hnsw->dist_rec.clear();

                // Select query
                switch(tt) {
                    case 0: // Run query in regular mode
                        result = hnsw->searchKnnDR(qs + i * dim, k);
                        break;
                    case 1: // Run query for greedy search then base layer search
                        result = hnsw->searchKnnDRgre(qs + i * dim, k);
                        break;
                    case 2: // Run query for numinit then base layer search
                        result = hnsw->searchKnnDRini(qs + i * dim, k, 50);
                        break;
                    case 3: // Run query for numinit then base layer search
                        result = hnsw->searchKnnDRini(qs + i * dim, k, 100);
                        break;
                    case 4: // Run query for numinit then base layer search
                        result = hnsw->searchKnnDRini(qs + i * dim, k, 200);
                        break;
                    case 5: // Run query for numinit then base layer search
                        result = hnsw->searchKnnDRini(qs + i * dim, k, 400);
                        break;
                }
            }
            const std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();
            // record data
            save_results(result, lat, dc, recall, numrun, i, k, gt, t1, t2, hnsw);
        }

        // save to file
        compute_and_save(dim, efc, m, efs, k, lat, dc, recall, qnum, numrun, resultsFP[tt]);
    }

    // by now results should have been saved to lat, distcomp, recall
    return 0;
}


int main() 
{
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

        std::vector<std::string> test_tag = {
            "reg-",
            "gre-",
            "ini50-",
            "ini100-",
            "ini200-",
            "ini400-",
        };
        FILE *resultsFP[6];
        // Create output file
        for (int tt = 0; tt < test_tag.size(); tt++) {
            std::string resultfile = std::string(HNSWOUTPATH) + MODELTAG /*"m7-"*/ + test_tag[tt] +
                                dataname[i][NAME] + 
                                ".txt";
            const char *resultfilec = resultfile.c_str();
            resultsFP[tt] = std::fopen(resultfilec, FILEMODE);

            #ifndef FILEAPP
                std::fprintf(resultsFP[tt], "%4s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%12s,%12s,%12s,%8s,%8s,%8s\n", 
                            "d", "efc", "m", "efs", "k", "lp50", "lp99", "lave", "dp50", "dp99", "dave", "rp50", "rp99", "rave");
            #endif
        }

        // Build index
        std::cout << "\tInitializing index..." << std::endl;

        // Define distance 
        hnswlib::SpaceInterface<float> *distance = nullptr;
        if(dataname[i][SPACE] == "L2") {
            std::cout << "\tCreating L2 space..." << std::endl;
            distance = new hnswlib::L2Space(metadata[i][DIM]);
        }
        else {
            std::cout << "\tCreating IP space..." << std::endl;
            distance = new hnswlib::InnerProductSpace(metadata[i][DIM]);
        }
        // Declare hnsw
        hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(
            distance,               // SpaceInterface<dist_t> *s
            metadata[i][DATASIZE],  // size_t max_elements,
            metadata[i][M],         // size_t M = 16,
            metadata[i][EFC],       // size_t ef_construction = 200,
            100                    // size_t random_seed = 100
        );
        // Add data to index
        std::cout << "\tAdding points..." << std::endl;
        for (int j = 0; j < metadata[i][DATASIZE]; j++) {
            float* point_data = data + j * metadata[i][DIM];
            hnsw->addPoint(point_data, j);
        }

        // Run tests with increasing beam width
        std::cout << "\tStarting benchmarking for " << dataname[i][NAME] << "...\n";
        // Setup
        int testnum = 1;
        // For each efs
        for (int efsind=0; efsind < EFSNUM; efsind++) {

            std::cout << "\tRunning test " << testnum << "...\n";

            hnsw->setEf(efsarr[efsind]);

            int qnum = metadata[i][QNUM];

            run_hnsw(metadata[i][DIM],                  // dim 
                    metadata[i][EFC],                   // efc
                    metadata[i][M],                     // m
                    efsarr[efsind],                     // efs
                    metadata[i][GTNUM],                 // k
                    qnum,                               // qnum
                    qs, gt,                             // input arrays
                    hnsw,                               // hnsw
                    resultsFP,                          // files to write to
                    test_tag                            // file tags
            ); 

            // #ifdef DEBUG
            //     std::vector<float> recall1(qnum, 0);
            //     compute_recall(recall1, matrix1, gt, qnum, metadata[i][GTNUM]);

            //     std::vector<float> recall2(qnum, 0);
            //     compute_recall(recall2, matrix2, gt, qnum, metadata[i][GTNUM]);

            //     if (efsind == EFSNUM - 1) {
            //         std::string sampleName = std::string(HNSWOUTPATH) + 
            //                 MODELTAG /*"m5-"*/ + 
            //                 dataname[i][NAME] + 
            //                 "-M" + mstring + 
            //                 OPTFLAG/*"-opt"*/ + "-sample.txt";
            //         const char* sampleCName = sampleName.c_str();
            //         FILE *sample = std::fopen(sampleCName, "w");

            //         std::fprintf(sample, "%4s%8s%8s%8s%8s%8s%8s%8s\n", "i", "m1", "m2", "m", "o", "dc1", "dc2", "dc");
            //         for (int r = 0; r < qnum; r++)
            //             std::fprintf(sample, "%4d%8.2f%8.2f%8.2f%8.2f%8d%8d%8d\n", r, recall1[r], recall2[r], recall[r], overlap[r], distcomp1[r], distcomp2[r], distcomp1[r]+distcomp2[r]);

            //         fclose(sample);
            //     }
            // #endif
            
            testnum++;
        }

        // Clean up graph
        delete distance;
        delete hnsw;

        // Clean up data
        delete[] data;
        delete[] qs;
        delete[] gt;
        data = nullptr;
        qs = nullptr;
        gt = nullptr;

        // close files
        for (int tt = 0; tt < test_tag.size(); tt++) {
            std::fclose(resultsFP[tt]);
        }

    } // for each dataset

    return 0;
}