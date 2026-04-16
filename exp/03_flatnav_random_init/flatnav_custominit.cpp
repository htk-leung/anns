/*
  To run :
    cd flatnav
    ./bin/build.sh -e
    cd build
    ./flatnav_data

  WORKS DON'T TOUCH
*/
#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include "../../code/helpers.h"
#include <iostream>
#include <chrono>
#include <random>
#include <cstdint>

// #define resultfile "../../logresult/20251002-flatnav-gen-s200-c200.txt"
// #define DATAFILENAME "../../data/rand/random-10000-dim-128.bin"
// #define QSFILENAME "../../data/rand/random-10000-dim-128-qs100.bin"
// #define GTFILENAME "../../data/rand/random-10000-dim-128-gt100.bin"

#define resultfile "../../logresult/20251007-flatnav-sift-s10c100.txt"
#define DATAFILENAME "../../data/sift1m/sift10k_base.bin"
#define QSFILENAME "../../data/sift1m/sift10k_query100.bin"
#define GTFILENAME "../../data/sift1m/sift10k_groundtruth100.bin"

int snuminit = 10, cnuminit = 100;

#define NUMTHREAD 1
#define DIM 128
#define DATASIZE 10000
#define QNUM 100
#define GTNUM 100


template <typename dist_t>
float run_knn_search(flatnav::Index<dist_t, int>* index, float *qs, int* gt, int efs, int k, int gtnum, int qnum, int dim, int snuminit) {

  float recall = 0;
  float correct = 0;
  for (int i = 0; i < qnum; i++) {            // for each query item
    float *q = qs + dim * i;                  // q = this item
    int *g = gt + gtnum * i;                 // g = ground truth set for this item. *(gt + i * K + j)
                                              // need to separate k and num_gt because num_gt is set in incoming dataset gt, but k can vary

    std::vector<std::pair<float, int>> result = index->search(q, k, efs,
                                                              /*num_initializations = 100*/ snuminit); // list of NN

    for (int j = 0; j < k; j++) {             // for each of the result
      for (int l = 0; l < k; l++) {           // find match in ground truth array for the current query item 
        // std::cout << "result[" << j << "].second : " << result[j].second << "\tg[" << l << "] : " << g[l] << std::endl;
        if (result[j].second == g[l]) {
          correct++;
        }
      }
    }
  }
  return recall = correct / (k * qnum);
}

float recall_flatnav(int testnum, int dim, float *data, float *qs, int *gt, int datasize, int qnum, int gtnum,
  int m, int efc, int efs, int k, int snuminit, int cnuminit, std::ofstream& result) {
    
    // Create an index with l2 distance 
    auto distance = flatnav::distances::SquaredL2Distance<>::create(dim);
    auto* index = new flatnav::Index<flatnav::distances::SquaredL2Distance<DataType::float32>, int>(
                          /* dist = */ std::move(distance), 
                          /* dataset_size = */ datasize,
                          /* max_edges_per_node = */ m,
                          /* collect_stats */ true);

    index->setNumThreads(NUMTHREAD);

    std::vector<int> labels(datasize);
    std::iota(labels.begin(), labels.end(), 0);

    index->addBatch<float>(/* data = */ (void *)data,
                                    /* labels = */ labels,
                                    /* ef_construction */ efc,
                                    /*num_initializations = 100*/ cnuminit);
    
    // get distance computation count and reset parameters for search count
    uint64_t constr_dist_comp = index->distanceComputations();
    index->resetStats();

    // query the index and compute the recall. 
    float recall = run_knn_search(index, qs, gt, efs, k, gtnum, qnum, dim, snuminit);

    uint64_t search_dist_comp = index->distanceComputations();

    result << dim << "," << efc << "," << m << "," << efs << "," << k << "," << recall << "," << constr_dist_comp << "," << search_dist_comp << std::endl;

    std::cout << "flatnav kNN Test " << testnum << ":\nRecall: " << recall << "\n";

    delete index;
    return recall;
}

int main(int argc, char **argv) {
    // create result file
    std::ofstream result(resultfile, std::ios::out); 

    result << "d,efc,m,efs,k,recall,distcountconstr,distcountsearch,distcache" << std::endl;

    // import data
    std::ifstream datafile(DATAFILENAME, std::ios::binary);
    std::ifstream qsfile(QSFILENAME, std::ios::binary);
    std::ifstream gtfile(GTFILENAME, std::ios::binary);

    if(!datafile | !qsfile | !gtfile)
      throw std::runtime_error("Failed to open gt file.\n");

    float *data = nullptr, *qs = nullptr;
    int *gt = nullptr;

    try {
        // datatype *bin_to_arr(std::ifstream& file, std::string filename, int arrsize)
        data = bin_to_arr<float>(datafile, DATAFILENAME, DATASIZE);
        qs = bin_to_arr<float>(qsfile, QSFILENAME, QNUM);
        gt = bin_to_arr<int>(gtfile, GTFILENAME, QNUM);

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

    // run tests
    // for now, keep efs = k to align with hnsw
    int testnum = 1;
    for (int efs = 100; efs <= 400; efs*=2) // k
    {
        for (int m = 32; m <=128; m*=2)
        {
            for (int efc = 50; efc <= 200; efc *= 2)
            {
              recall_flatnav(testnum++, DIM, data, qs, gt, DATASIZE, QNUM, GTNUM, m, efc, efs, GTNUM, snuminit, cnuminit, result);
            }
        }
    }
    // int m = 32, efc = 50, efs = 50, d = 128, datasize = DATASIZE;
    // recall_flatnav(1, DIM, data, qs, gt, DATASIZE, QNUM, GTNUM, m, efc, efs, GTNUM, result);

    result.close();

    delete[] data;
    delete[] qs;
    delete[] gt;

    return 0;
}