/*
    dim             // space dimension
    max_elements    // Maximum number of elements, should be known beforehand
    M               // Tightly connected with internal dimensionality of the data
                    // strongly affects the memory consumption
    ef_construction // Controls index search speed/build speed tradeoff
    k
*/

#include "../hnswlib/hnswlib/hnswlib.h"
#include "helpers.h"
#include "helpers_hnsw.h"
#include <fstream>
#include <chrono>

#define logfile "../logresult/20250926-hnsw-rand.log"
#define resultfile "../logresult/20250926-hnsw-rand.txt"

#define DATAFILENAME16 "../data/rand/random-10000-dim-16.bin"
#define QSFILENAME16 "../data/rand/random-10000-dim-16-qs100.bin"
#define GTFILENAME16 "../data/rand/random-10000-dim-16-gt100.bin"
#define DATAFILENAME32 "../data/rand/random-10000-dim-32.bin"
#define QSFILENAME32 "../data/rand/random-10000-dim-32-qs100.bin"
#define GTFILENAME32 "../data/rand/random-10000-dim-32-gt100.bin"
#define DATAFILENAME64 "../data/rand/random-10000-dim-64.bin"
#define QSFILENAME64 "../data/rand/random-10000-dim-64-qs100.bin"
#define GTFILENAME64 "../data/rand/random-10000-dim-64-gt100.bin"

#define DATASIZE 10000
#define NUMQ 100  // must be <= number of queries in dataset qs
#define NUMGT 100 // used to skip to gt for CURRENT query element
                  // should match K in create_data_random_gt.cpp for gt

void recall_rand(int testnum, int dim, int max_elements, int m, int ef_construction, int k, int numq, 
    float *data, float *qs, int *gt, std::ofstream& log, std::ofstream& result);

int main() {
    // create log in append mode
    std::ofstream log(logfile, std::ios::out); 
    std::ofstream result(resultfile, std::ios::out); 

    result << "d,efc,m,efs,k,recall,searchtime" << std::endl;

    // import data
    std::ifstream datafile16(DATAFILENAME16, std::ios::in);
    std::ifstream qsfile16(QSFILENAME16, std::ios::in);
    std::ifstream gtfile16(GTFILENAME16, std::ios::in);

    std::ifstream datafile32(DATAFILENAME32, std::ios::in);
    std::ifstream qsfile32(QSFILENAME32, std::ios::in);
    std::ifstream gtfile32(GTFILENAME32, std::ios::in);

    std::ifstream datafile64(DATAFILENAME64, std::ios::in);
    std::ifstream qsfile64(QSFILENAME64, std::ios::in);
    std::ifstream gtfile64(GTFILENAME64, std::ios::in);

    if(!datafile16 | !datafile32 | !datafile64)
      throw std::runtime_error("Failed to open data file.\n");
    if(!qsfile16 | !qsfile32 | !qsfile64)
      throw std::runtime_error("Failed to open queries file.\n");
    if(!gtfile16 | !gtfile32 | !gtfile64)
      throw std::runtime_error("Failed to open gt file.\n");
      
    float *data16 = bin_to_arr<float>(datafile16, DATAFILENAME16);
    float *qs16 = bin_to_arr<float>(qsfile16, QSFILENAME16);
    int *gt16 = bin_to_arr<int>(gtfile16, GTFILENAME16);

    float *data32 = bin_to_arr<float>(datafile32, DATAFILENAME32);
    float *qs32 = bin_to_arr<float>(qsfile32, QSFILENAME32);
    int *gt32 = bin_to_arr<int>(gtfile32, GTFILENAME32);

    float *data64 = bin_to_arr<float>(datafile64, DATAFILENAME64);
    float *qs64 = bin_to_arr<float>(qsfile64, QSFILENAME64);
    int *gt64 = bin_to_arr<int>(gtfile64, GTFILENAME64);

    float *data = nullptr;
    float *qs = nullptr;
    int *gt = nullptr;

    datafile16.close();
    qsfile16.close();
    gtfile16.close();

    datafile32.close();
    qsfile32.close();
    gtfile32.close();

    datafile64.close();
    qsfile64.close();
    gtfile64.close();

    // recall_efs_k(int testnum, int dim, int max_elements, int M, int ef_construction, int k, int numtest, std::ofstream&);
    int testind = 1;
    
    for (int efs = 10; efs <= 80; efs*=2)
    {
        for (int m = 16; m <= 128;m*=2)
        {
            for (int efc = 50; efc <= 200; efc*=2)
            {
                for (int d = 16; d <= 128; d*=2)
                {
                    switch(d) {
                        case 16:
                        data = data16;
                        qs = qs16;
                        gt = gt16;
                        break;
                        case 32:
                        data = data32;
                        qs = qs32;
                        gt = gt32;
                        break;
                        case 64:
                        data = data64;
                        qs = qs64;
                        gt = gt64;
                        break;
                        default:
                        data = data64;
                        qs = qs64;
                        gt = gt64;
                        break;
                    } 
                    recall_rand(testind++, d, DATASIZE, m, efc, efs, NUMQ, data, qs, gt, log, result);
                }
            }
        }
    }

    return 0;
}

// this returns recall for kNN
// find kNN for each and average out percentage
void recall_rand(int testnum, int dim, int max_elements, int m, int ef_construction, int k, int numq, 
    float *data, float *qs, int *gt, std::ofstream& log, std::ofstream& result) {

    log << "recall_efs_k Test " << testnum << std::endl;
    log << "\t max_elements = " << max_elements
        << "\tdim = " << dim
        << "\tef_construction = " << ef_construction 
        << "\tM = " << m
        << "\tef_search = " << k
        << "\tk = " << k << std::endl;

    auto starttime = std::chrono::high_resolution_clock::now();

    // Initing index | EDIT : alg_brute to find ground-truth kNN
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, m, ef_construction);

    // Add data to index | EDIT : add points to alg_brute
    for (int i = 0; i < max_elements; i++) {
        float* point_data = data + i * dim;
        alg_hnsw->addPoint(point_data, i);
    }

    auto endtime = std::chrono::high_resolution_clock::now();
    auto duration = (endtime - starttime).count(); 
    log << "\tConstruction time\t: " << duration << "ms" << std::endl;
    starttime = std::chrono::high_resolution_clock::now();

    // Query the elements for themselves and measure recall | EDIT : change recall calculation
    // try to find the first numtest items
    float correct = 0;
    for (int i = 0; i < numq; i++) {
        float *q = qs + dim * i;
        int *g = gt + k * i;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(q, k);

        // recall
        int val;
        for (int j = 0; j < numq; j++) {                 // for each of the result
            val = result.top().second;
            result.pop();
            for (int l = 0; l < k; l++) {      // find match in ground truth array for the current query item 
                // std::cout << "result[" << j << "].second : " << result[j].second << "\tg[" << l << "] : " << g[l] << std::endl;
                if (val == g[l]) {
                    correct++;
                    break;
                }
            }
        }
    }

    float recall = correct / (numq * k);

    endtime = std::chrono::high_resolution_clock::now();
    duration = (endtime - starttime).count(); 
    log << "\tSearch time\t\t\t: " << duration << "ms" << std::endl;
    log << "\tRecall\t\t\t\t: " << recall << std::endl << std::endl;
    result << dim << "," << ef_construction << "," << m << k << "," << k << "," << recall << "," << duration << std::endl;     //"d,efc,m,efs,k,recall"

    std::cout << "kNN Test " << testnum << ":\nRecall: " << recall << std::endl;

    delete alg_hnsw;
}