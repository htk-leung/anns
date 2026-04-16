/*
    This code is the first pass in running multiple hnsw graphs and combining results

    - for each graph
    - build with M not M/2, others keep constant
    - search with same set of efs

    (1) kNN OVERLAP
    - for each layer
    - save nodes visited by index #
    - for each layer calculate % overlap between 2 graphs

    (2) kNN
    V combine results to calculate recall >> together what do they achieve
    V kNN overlap


    (3) MULTIPLE ENTRY
    - build 1 graph
    - select top3
    - each gets a narrower beam??

    ** This is the fourth version where insertion order is shuffled.
    ** Results in lower overlap between kNN results from each graph and higher recall rates

*/

#include <fstream>
#include <chrono>

#include "../../hnswlib/hnswlib/hnswalg.h"
#include "../../src/utils.h"
#include "../../src/datadef.h"
#include "../../src/outdef.h"
#include "../helpers_hnsw.h"


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


// Function returns recall for kNN search as average recall percentage
float run_hnsw(int dim, int k, int qnum, int dcbudget,
    float *qs,  
    std::vector<float>& lat,
    std::vector<int>& distcomp,
    std::vector<std::vector<hnswlib::labeltype>>& matrix,
    hnswlib::HierarchicalNSW<float>* alg_hnsw) 
{
    // Query setup
    int numrun = 2; /* number of times to run queries to average out unexpected performance disturbances */

    // For each query point
    for (int i = 0; i < qnum; i++) {
        // Clean up stats
        alg_hnsw->resetMetrics();

        // Run query
        const std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
        for (int r = 0; r < numrun; r++)
            result = alg_hnsw->searchKnnMulGraph(dcbudget, qs + i * dim, k);
        const std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();

        // copy labels to vector
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
        distcomp[i] = alg_hnsw->metric_distance_computations / double(numrun); //

        // Hops
        // hops[i] = alg_hnsw->metric_hops / double(numrun);
        
    }
    return 0;
}

void combine_matrices(std::vector<std::vector<hnswlib::labeltype>>& m, 
    std::vector<std::vector<hnswlib::labeltype>>& m1, 
    std::vector<std::vector<hnswlib::labeltype>>& m2, 
    std::vector<float>& overlap,
    int qnum, int k) 
{
    std::cout << "\t\t\tCombining matrices...\n";

    int j1, j2, o;
    
    for (int i = 0; i < qnum; i++) { // for each row
        std::vector<hnswlib::labeltype> row;

        j1 = 0;
        j2 = 0;
        o = 0;

        // combine
        while ( j1 < k && j2 < k ) { // run through array with smaller max value
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
        if (j1 < k) { // run through the rest of the other array
            while (j1 < k)
                row.push_back(m1[i][j1++]);
        }
        if (j2 < k) {
            while (j2 < k)
                row.push_back(m2[i][j2++]);
        }
        // if they don't end at k there's a problem
        if (j1!=k || j2!=k)
            throw std::runtime_error("Error combining matrices");

        // convert to rate of overlap
        std::cout << "o: " << overlap[i];
        std::cout << " " << o;
        overlap[i] = float(o) / k;
        std::cout << " " << overlap[i] << std::endl;

        // save and update j for next loop
        m[i] = row;
        std::cout << "m: " << m[i].size() << std::endl
                << "m1: " << m1[i].size() << std::endl
                << "m2: " << m2[i].size() << std::endl;
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
    std::vector<float>& lat2,
    std::vector<int>& distcomp2,
    std::vector<float>& overlap,
    std::vector<float>& recall,
    int qnum, int i, int testnum,
    std::ofstream& result) 
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

    // Print
    std::cout << "\t\t\tRecall: " << rave << std::endl;

    // Save to result file
    result << dim << "," << efc << "," << m << "," << efs << "," << k << "," << dc << ","
            << lp50 << "," << lp99 << "," << lave << "," 
            << dp50 << "," << dp99 << "," << dave << "," 
            << op50 << "," << op99 << "," << oave << "," 
            << rp50 << "," << rp99 << "," << rave
            << std::endl;
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

        // Build index
        std::cout << "\tInitializing index..." << std::endl;
        hnswlib::HierarchicalNSW<float>* alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(
            distance,               // SpaceInterface<dist_t> *s
            metadata[i][DATASIZE],  // size_t max_elements,
            metadata[i][M],         // size_t M = 16,
            metadata[i][EFC],       // size_t ef_construction = 200,
            100                     // size_t random_seed = 100
        );
        hnswlib::HierarchicalNSW<float>* alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(
            distance,               // SpaceInterface<dist_t> *s
            metadata[i][DATASIZE],  // size_t max_elements,
            metadata[i][M],         // size_t M = 16,
            metadata[i][EFC],       // size_t ef_construction = 200,
            200                     // size_t random_seed = 100
        );

        // Add data to index
        std::cout << "\tAdding points..." << std::endl;

        std::vector<int> indices(metadata[i][DATASIZE]);
        std::iota(indices.begin(), indices.end(), 0);

        std::default_random_engine rng2(200);
        std::shuffle(indices.begin(), indices.end(), rng2);

        for (int n = 0; n < metadata[i][DATASIZE]; n++) {
            float* point_data = data + n * metadata[i][DIM];
            alg_hnsw1->addPoint(point_data, n);

            point_data = data + indices[n] * metadata[i][DIM];
            alg_hnsw2->addPoint(point_data, indices[n]);
        }
        

        // Results
        int qnum = metadata[i][QNUM];

        std::vector<float> lat1(qnum);
        std::vector<int> distcomp1(qnum);
        std::vector<std::vector<hnswlib::labeltype>> matrix1(qnum);

        std::vector<float> lat2(qnum);
        std::vector<int> distcomp2(qnum);
        std::vector<std::vector<hnswlib::labeltype>> matrix2(qnum);

        std::vector<std::vector<hnswlib::labeltype>> matrix(qnum);
        std::vector<float> overlap(qnum, 0.0);
        std::vector<float> recall(qnum, 0);


        // Create output file
        std::string resultfile = std::string(HNSWOUTPATH) + "m4-" +
                                dataname[i][NAME] + OPTFLAG + ".txt";
        std::ofstream result(resultfile, std::ios::MODE);

        #ifndef FILEAPP
            result  << "d,efc,m,efs,k,dc,"
                    << "lp50,lp99,lave, "
                    << "dp50,dp99,dave," 
                    << "op50,op99,oave," 
                    << "rp50,rp99,rave"
                    << std::endl;
        #endif

        // Run tests with increasing beam width
        std::cout << "\tStarting benchmarking for " << dataname[i][NAME] << "...\n";

        // Setup
        int testnum = 1;

        for (int efsind=0; efsind < EFSNUM; efsind++) {

            // set dc budget for this run
            int budget = dccount[i][efsind] / 2;

            std::cout << "\tRunning test " << testnum << "...\n";

            alg_hnsw1->setEf(efsarr[efsind]);
            alg_hnsw2->setEf(efsarr[efsind]);

            run_hnsw(metadata[i][DIM],                  // dim
                    metadata[i][GTNUM],                 // k
                    qnum,                               // qnum
                    budget,                             // dist comp budget
                    qs,                                 // input arrays
                    lat1, distcomp1, matrix1,           // output
                    alg_hnsw1                           // hnsw
            ); 

            run_hnsw(metadata[i][DIM],                  // dim
                    metadata[i][GTNUM],                 // k
                    qnum,                               // qnum
                    budget,                             // dist comp budget
                    qs,                                 // input arrays
                    lat2, distcomp2, matrix2,           // output
                    alg_hnsw2                           // hnsw
            ); 

            // combine items into 1 array
            combine_matrices(matrix, matrix1, matrix2, overlap, qnum, metadata[i][GTNUM]);

            // For each item in combined, find it in gt
            compute_recall(recall, matrix, gt, qnum, metadata[i][GTNUM]);

            // Compute and save to file 
            compute_and_save(metadata[i][DIM], metadata[i][EFC], metadata[i][M] / 2, efsarr[efsind], metadata[i][GTNUM], budget, 
                lat1, distcomp1, lat2, distcomp2, overlap, recall, qnum, i, testnum, result);

            testnum++;
        }
        
        // Clean up
        delete[] data;
        delete[] qs;
        delete[] gt;
        delete alg_hnsw1;
        delete alg_hnsw2;
        delete distance;

        // Reset pointers
        data = nullptr;
        qs = nullptr;
        gt = nullptr;
    }

    return 0;
}