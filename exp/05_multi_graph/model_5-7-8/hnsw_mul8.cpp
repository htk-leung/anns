/*
    This code is the first pass in running multiple hnsw graphs and combining results

    M8 implements:

    - split model search (2 hnsw graphs)
    - dcbudget for graph1 and graph2 split 1.5 / 0.5
    - overall knn overlap
    - dcbudget defined from normal hnsw search run
    - shared distance caching between graphs s.t. distances calculated from graph1 is reused in graph2
    - cache miss for each graph

    M7a: without insertion order shuffling
    M7b: with
*/

#include <fstream>
#include <chrono>

#include "../../hnswlib/hnswlib/hnswalg.h"
#include "../../hnswlib/hnswlib/hnswm.h"
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

#define MODELTAG "m8-"

// only written for numrun = 2, division logic won't work otherwise
void update_latency( // update_latency(lat1, numrun, i, t1, t2);
    std::vector<float>& lat,
    int& numrun, int& i, 
    const std::chrono::time_point<std::chrono::high_resolution_clock> t1,
    const std::chrono::time_point<std::chrono::high_resolution_clock> t2) 
{   
    // Latency
    double latency = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
    lat[i] = (lat[i] + latency) / double(numrun); //
}
void save_results(std::priority_queue<std::pair<float, hnswlib::labeltype>>& result, 
    std::vector<std::vector<hnswlib::labeltype>>& matrix,
    std::vector<float>& lat,
    std::vector<int>& distcomp,
    std::vector<float>& cachemiss,
    int& numrun, int& i, int& k, int& hnswnewvisitcount,
    const std::chrono::time_point<std::chrono::high_resolution_clock> t1,
    const std::chrono::time_point<std::chrono::high_resolution_clock> t2,
    hnswlib::HierarchicalNSW<float>* alg_hnsw) 
{   
    // Copy labels to vector
    std::vector<hnswlib::labeltype> knnlabels;
    knnlabels.reserve(k);
    while(!result.empty()) {
        knnlabels.push_back(result.top().second);
        result.pop();
    }

    // sort and save
    std::sort(knnlabels.begin(), knnlabels.end());
    matrix[i] = knnlabels;
    
    // Latency
    double latency = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
    lat[i] = latency / double(numrun); //
    // Dist Comp
    distcomp[i] = alg_hnsw->metric_distance_computations;
    // Cache miss
    int hnswdistfunccalls = alg_hnsw->metric_distfunc_calls;
    cachemiss[i] = float(hnswnewvisitcount) / hnswdistfunccalls;
}

// Function returns recall for kNN search as average recall percentage
float run_hnsw(int dim, int k, int qnum, int dcbudget,
    float *qs,  
    std::vector<float>& lat1,
    std::vector<int>& distcomp1,
    std::vector<float>& cachemiss1,
    std::vector<std::vector<hnswlib::labeltype>>& matrix1,
    std::vector<float>& lat2,
    std::vector<int>& distcomp2,
    std::vector<float>& cachemiss2,
    std::vector<std::vector<hnswlib::labeltype>>& matrix2,
    hnswlib::hnswM<float>* hnswSet) 
{
    // Query setup
    int numrun = 2; /* number of times to run queries to average out unexpected performance disturbances */

    // For each query point
    for (int i = 0; i < qnum; i++) {
        for (int n = 0; n < numrun; n++) {
            // Clear metrics
            hnswSet->hnsw1->resetMetrics();
            hnswSet->hnsw2->resetMetrics();
            hnswSet->resetCache();

            // Run query in graph 1
            const std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result1;
            result1 = hnswSet->hnsw1->searchKnnMC(dcbudget * 1.5, qs + i * dim, k);
            const std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();

            int hnsw1newvisitcount = hnswSet->dist_cache_->size();
            int hnsw1distfunccalls = hnswSet->hnsw1->metric_distfunc_calls;

            // Run the same query in graph 2
            const std::chrono::time_point<std::chrono::high_resolution_clock> t3 = std::chrono::high_resolution_clock::now();
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result2;
            result2 = hnswSet->hnsw2->searchKnnMC(dcbudget * 0.5, qs + i * dim, k);
            const std::chrono::time_point<std::chrono::high_resolution_clock> t4 = std::chrono::high_resolution_clock::now();

            int hnsw2newvisitcount = hnswSet->dist_cache_->size() - hnsw1newvisitcount;
            int hnsw2distfunccalls = hnswSet->hnsw2->metric_distfunc_calls;

            // save dist comp and latency for graph 2 for this query
            if (n == 0) {
                save_results(result1, matrix1, lat1, distcomp1, cachemiss1, numrun, i, k, hnsw1newvisitcount, t1, t2, hnswSet->hnsw1);
                save_results(result2, matrix2, lat2, distcomp2, cachemiss2, numrun, i, k, hnsw2newvisitcount, t3, t4, hnswSet->hnsw2); 
            }
            else {
                update_latency(lat1, numrun, i, t1, t2);
                update_latency(lat2, numrun, i, t3, t4);
            }
        }
    }

    // by now results should have been saved to matrix1, lat1, distcomp1, matrix2, lat2, distcomp2
    return 0;
}

void combine_matrices(std::vector<std::vector<hnswlib::labeltype>>& m, 
    std::vector<std::vector<hnswlib::labeltype>>& m1, 
    std::vector<std::vector<hnswlib::labeltype>>& m2, 
    std::vector<float>& overlap,
    int qnum, int k) 
{
    std::cout << "\t\t\tCombining matrices...\n";

    int j1, j2, o, k1, k2;
    
    for (int i = 0; i < qnum; i++) { // for each row
        std::vector<hnswlib::labeltype> row;

        j1 = 0;
        j2 = 0;
        o = 0;
        k1 = m1[i].size();
        k2 = m2[i].size();
        // #ifdef DEBUG
        //     std::cout<< "k1: " << k1 << "\t\tk2: " << k2 << std::endl;
        // #endif

        // combine
        while ( j1 < k1 && j2 < k2 ) { // run through array with smaller max value
            if (m1[i][j1] == m2[i][j2]) { // if both have the same value, just add it once and both can move on
                row.push_back(m1[i][j1]);
                o++;
                j1++;
                j2++;
            }
            else { // if values are different, add only the smaller value and increment j so that it can 'catch up' with the other one
                if (m1[i][j1] < m2[i][j2]) {
                    row.push_back(m1[i][j1]);
                    j1++;
                }
                else {
                    row.push_back(m2[i][j2]);
                    j2++;
                }
            }
        }
        if (j1 < k1) { // run through the rest of the other array
            while (j1 < k1)
                row.push_back(m1[i][j1++]);
        }
        if (j2 < k2) {
            while (j2 < k2)
                row.push_back(m2[i][j2++]);
        }
        // if they don't end at k there's a problem
        if (j1!=k1 || j2!=k2)
            throw std::runtime_error("Error combining matrices");

        // convert to rate of overlap
        #ifdef DEBUG
            // std::cout << "o: " << overlap[i];
            // std::cout << " " << o;
        #endif

        overlap[i] = float(o) / k;

        #ifdef DEBUG
            // std::cout << " " << overlap[i] << std::endl;
        #endif

        // save and update j for next loop
        m[i] = row;
        
        #ifdef DEBUG
            // std::cout << "m: " << m[i].size() << std::endl
            //         << "m1: " << m1[i].size() << std::endl
            //         << "m2: " << m2[i].size() << std::endl;
        #endif
    }
}

void compute_recall(std::vector<float>& recall, 
    std::vector<std::vector<hnswlib::labeltype>>& m, 
    int *gt, int qnum, int k) 
{
    std::cout << "\t\t\tComputing recall...\n";

    int s = 0, r;
    // for each query
    for (int i = 0; i < qnum; i++) { 
        s = m[i].size();
        r = 0;

        // for each thing found
        for (int j = 0; j < s; j++) {
            // loop through gt to find match
            for (int n = 0; n < k; n++) {
                if (m[i][j] == gt[i * k + n]) {
                    r++;
                    break;
                }
            }
        }
        recall[i] = float(r) / k;
    }        
}

void compute_and_save(
    int dim, int efc, int m, int efs, int k, int dc,
    std::vector<float>& lat1,
    std::vector<int>& distcomp1,
    std::vector<float>& cachemiss1,
    std::vector<float>& lat2,
    std::vector<int>& distcomp2,
    std::vector<float>& cachemiss2,
    std::vector<float>& overlap,
    std::vector<float>& recall,
    int qnum, int i, int testnum,
    FILE* result) 
{
    std::cout << "\t\t\tComputing the rest and saving...\n";
    
    // Get
    std::sort(lat1.begin(), lat1.end());
    std::sort(lat2.begin(), lat2.end());
    float lp50 = percentile<float>(lat1, 50, qnum) + percentile<float>(lat2, 50, qnum);
    float lp99 = percentile<float>(lat1, 99, qnum) + percentile<float>(lat2, 99, qnum);
    float lave = average<float>(lat1) + average<float>(lat2);

    std::sort(distcomp1.begin(), distcomp1.end());
    std::sort(distcomp2.begin(), distcomp2.end());
    float dp50 = percentile<int>(distcomp1, 50, qnum) + percentile<int>(distcomp2, 50, qnum);
    float dp99 = percentile<int>(distcomp1, 99, qnum) + percentile<int>(distcomp2, 99, qnum);
    float dave = average<int>(distcomp1) + average<int>(distcomp2);

    std::sort(overlap.begin(), overlap.end());
    float op50 = percentile<float>(overlap, 50, qnum);
    float op99 = percentile<float>(overlap, 99, qnum);
    float oave = average<float>(overlap);

    std::sort(recall.begin(), recall.end());
    float rp50 = percentile<float>(recall, 50, qnum);
    float rp99 = percentile<float>(recall, 99, qnum);
    float rave = average<float>(recall);

    std::sort(cachemiss1.begin(), cachemiss1.end());
    float c1p50 = percentile<float>(cachemiss1, 50, qnum);
    float c1p99 = percentile<float>(cachemiss1, 99, qnum);
    float c1ave = average<float>(cachemiss1);

    std::sort(cachemiss2.begin(), cachemiss2.end());
    float c2p50 = percentile<float>(cachemiss2, 50, qnum);
    float c2p99 = percentile<float>(cachemiss2, 99, qnum);
    float c2ave = average<float>(cachemiss2);

    // Print
    std::cout << "\t\t\tRecall: " << rave << std::endl;

    // Save to result file
    std::fprintf(result, "%4d,%8d,%8d,%8d,%8d,%8d,%8.2f,%8.2f,%8.2f,%12.2f,%12.2f,%12.2f,%8.2f,%8.2f,%8.2f,%8.2f,%8.2f,%8.2f,%8.2f,%8.2f,%8.2f,%8.2f,%8.2f,%8.2f\n", 
        dim, efc, m, efs, k, dc, lp50, lp99, lave, dp50, dp99, dave, op50, op99, oave, rp50, rp99, rave, c1p50, c1p99, c1ave, c2p50, c2p99, c2ave);
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


        // Results
        int qnum = metadata[i][QNUM];

        std::vector<float> lat1(qnum);
        std::vector<int> distcomp1(qnum);
        std::vector<float> cachemiss1(qnum);
        std::vector<std::vector<hnswlib::labeltype>> matrix1(qnum);

        std::vector<float> lat2(qnum);
        std::vector<int> distcomp2(qnum);
        std::vector<float> cachemiss2(qnum);
        std::vector<std::vector<hnswlib::labeltype>> matrix2(qnum);

        std::vector<std::vector<hnswlib::labeltype>> matrix(qnum);
        std::vector<float> overlap(qnum, 0.0);
        std::vector<float> recall(qnum, 0);

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

        // NEW SET
        hnswlib::hnswM<float>* hnswSet = new hnswlib::hnswM<float>();
        int m;
        // For each M build graphs
        for (int mInd = 0; mInd < M56_MNUM; mInd++ ) {

            m = metadata[i][M] / 2;
            // switch (mInd) {
            //     case 0:
            //         m = metadata[i][M];
            //         // efc = metadata[i][EFC] / 8;
            //         break;
            //     case 1:
            //         m = metadata[i][M] / 4;
            //         // efc = metadata[i][EFC] / 4;
            //         break;
            //     case 2:
            //         m = metadata[i][M] / 2;
            //         // efc = metadata[i][EFC] / 2;
            //         break;
            //     case 3:
            //         m = metadata[i][M];
            //         // efc = metadata[i][EFC];
            //         break;
            //     case 4:
            //         m = metadata[i][M] * 2;
            //         // efc = metadata[i][EFC] * 2;
            //         break;
            //     default:
            //         m = metadata[i][M];
            // }

            // Create output file
            std::string mstring = std::to_string(m);
            std::string resultfile = std::string(HNSWOUTPATH) + MODELTAG /*"m7-"*/ + 
                                    dataname[i][NAME] + 
                                    "-M" + mstring + 
                                    OPTFLAG/*"-opt"*/ + ".txt";
            const char *resultfilec = resultfile.c_str();
            FILE *result = std::fopen(resultfilec, FILEMODE);

            #ifndef FILEAPP
                std::fprintf(result, "%4s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%12s,%12s,%12s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s\n", 
                            "d", "efc", "m", "efs", "k", "dc", 
                            "lp50", "lp99", "lave", "dp50", "dp99", "dave", "op50", "op99", "oave", "rp50", "rp99", "rave", "c1p50", "c1p99", "c1ave", "c2p50", "c2p99", "c2ave");
            #endif

            // Build index
            std::cout << "\tInitializing index..." << std::endl;

            hnswSet->hnsw1 = new hnswlib::HierarchicalNSW<float>(
                distance,               // SpaceInterface<dist_t> *s
                metadata[i][DATASIZE],  // size_t max_elements,
                m,                      // size_t M = 16,
                metadata[i][EFC],       // size_t ef_construction = 200,
                100,                     // size_t random_seed = 100
                hnswSet->dist_cache_
            );

            hnswSet->hnsw2 = new hnswlib::HierarchicalNSW<float>(
                distance,               // SpaceInterface<dist_t> *s
                metadata[i][DATASIZE],  // size_t max_elements,
                m,                      // size_t M = 16,
                metadata[i][EFC],       // size_t ef_construction = 200,
                200,                    // size_t random_seed = 100
                hnswSet->dist_cache_
            );

            // Add data to index

            // std::cout << "\tAdding points..." << std::endl;
            // std::vector<int> indices(metadata[i][DATASIZE]);
            // std::iota(indices.begin(), indices.end(), 0);

            // std::default_random_engine rng2(200);
            // std::shuffle(indices.begin(), indices.end(), rng2);

            // for (int n = 0; n < metadata[i][DATASIZE]; n++) {
            //     float* point_data = data + n * metadata[i][DIM];
            //     hnswSet->hnsw1->addPoint(point_data, n);

            //     point_data = data + indices[n] * metadata[i][DIM];
            //     hnswSet->hnsw2->addPoint(point_data, indices[n]);
            // }

            std::cout << "\tAdding points..." << std::endl;
            for (int j = 0; j < metadata[i][DATASIZE]; j++) {
                float* point_data = data + j * metadata[i][DIM];
                hnswSet->hnsw1->addPoint(point_data, j);
                hnswSet->hnsw2->addPoint(point_data, j);
            }

            // Run tests with increasing beam width
            std::cout << "\tStarting benchmarking for " << dataname[i][NAME] << "...\n";

            // Setup
            int testnum = 1;

            // For each efs
            for (int efsind=0; efsind < EFSNUM; efsind++) {

                // set dc budget for this run
                int budget = dccount[i][efsind];

                std::cout << "\tRunning test " << testnum << "...\n";

                hnswSet->hnsw1->setEf(efsarr[efsind]);
                hnswSet->hnsw2->setEf(efsarr[efsind]);

                run_hnsw(metadata[i][DIM],                  // dim
                        metadata[i][GTNUM],                 // k
                        qnum,                               // qnum
                        budget,                             // dist comp budget
                        qs,                                 // input arrays
                        lat1, distcomp1, cachemiss1, matrix1,           // output
                        lat2, distcomp2, cachemiss2, matrix2,           // output
                        hnswSet                             // hnsw
                ); 

                // combine items into 1 array
                combine_matrices(matrix, matrix1, matrix2, overlap, qnum, metadata[i][GTNUM]);

                // For each item in combined, find it in gt
                compute_recall(recall, matrix, gt, qnum, metadata[i][GTNUM]);

                #ifdef DEBUG
                    std::vector<float> recall1(qnum, 0);
                    compute_recall(recall1, matrix1, gt, qnum, metadata[i][GTNUM]);

                    std::vector<float> recall2(qnum, 0);
                    compute_recall(recall2, matrix2, gt, qnum, metadata[i][GTNUM]);

                    if (efsind == EFSNUM - 1) {
                        std::string sampleName = std::string(HNSWOUTPATH) + 
                                MODELTAG /*"m5-"*/ + 
                                dataname[i][NAME] + 
                                "-M" + mstring + 
                                OPTFLAG/*"-opt"*/ + "-sample.txt";
                        const char* sampleCName = sampleName.c_str();
                        FILE *sample = std::fopen(sampleCName, "w");

                        std::fprintf(sample, "%4s%8s%8s%8s%8s%8s%8s%8s\n", "i", "m1", "m2", "m", "o", "dc1", "dc2", "dc");
                        for (int r = 0; r < qnum; r++)
                            std::fprintf(sample, "%4d%8.2f%8.2f%8.2f%8.2f%8d%8d%8d\n", r, recall1[r], recall2[r], recall[r], overlap[r], distcomp1[r], distcomp2[r], distcomp1[r]+distcomp2[r]);

                        fclose(sample);
                    }
                #endif

                // Compute and save to file 
                compute_and_save(metadata[i][DIM], metadata[i][EFC], m, efsarr[efsind], metadata[i][GTNUM], budget, 
                    lat1, distcomp1, cachemiss1, lat2, distcomp2, cachemiss2, overlap, recall, qnum, i, testnum, result);

                testnum++;
            }
            
            // Clean up graphs
            hnswSet->resetHnswSet();
            fclose(result);
        }
        // Clean up set
        delete distance;
        delete hnswSet;
        // Clean up data
        delete[] data;
        delete[] qs;
        delete[] gt;
        data = nullptr;
        qs = nullptr;
        gt = nullptr;
    }

    return 0;
}