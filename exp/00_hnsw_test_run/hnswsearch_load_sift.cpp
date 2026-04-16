/*
    This code calls the function 

        recall_data_k(...)

    to build hnsw index on the given dataset and calls functions

        xvec_get_dim    and
        xvec_to_arr

    to import fvec/ivec files as arrays.

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

#define logfile "../logresult/20250925-hnswsift.log"
#define resultfile "../logresult/20250925-hnswsift.txt"

#define DATAFILENAME "../data/sift/sift_base.fvecs"
#define GTFILENAME "../data/sift/sift_groundtruth.ivecs"
#define QSFILENAME "../data/sift/sift_query.fvecs"

#define DIM 128
#define DATASIZE 1000000
#define NUMQ 10000
#define K 100



int main() {
    // create log in append mode
    std::ofstream log(logfile, std::ios::out); 
    std::ofstream result(resultfile, std::ios::out); 

    result << "d,efc,m,efs,k,recall,searchtime" << std::endl;

    // load data
    std::ifstream datafile(DATAFILENAME, std::ios::binary);
    std::ifstream qsfile(QSFILENAME, std::ios::binary);
    std::ifstream gtfile(GTFILENAME, std::ios::binary);
    if(!datafile | !qsfile | !gtfile)
      throw std::runtime_error("Failed to open datafile in main.\n");

    // fvec/ivec to arrays
    float *data=nullptr, *qs=nullptr;
    int *gt=nullptr;

    try {
        // void xvec_to_arr(std::ifstream& file, datatype *outdata, int arrsize, int dim)
        xvec_to_arr<float>(datafile, &data, DATASIZE, DIM);
        xvec_to_arr<float>(qsfile, &qs, NUMQ, DIM);
        xvec_to_arr<int>(gtfile, &gt, NUMQ, K);

        // Verify we got valid pointers
        if (data == nullptr || qs == nullptr || gt == nullptr) {
            throw std::runtime_error("Pointers are still null after allocation");
        }

        // std::cout << "data : ";
        // for (int i = 0; i < 100; i++)
        //     std::cout << data[i] << " "; 
        // std::cout << std::endl << "qs : ";
        // for (int i = 0; i < 100; i++)
        //     std::cout << qs[i] << " "; 
        // std::cout << std::endl << "gt : ";
        // for (int i = 0; i < 100; i++)
        //     std::cout << gt[i] << " "; 
        // std::cout;

        datafile.close();
        qsfile.close();
        gtfile.close();
    } 
    catch (const std::exception& e) {
        std::cout << "Error : " << e.what() << std::endl;

        delete[] data;
        delete[] qs;
        delete[] gt;
        return 1;
    }

    // tests
    int testind = 1;
    
    for (int efs = 10; efs <= 80; efs*=2)
    {
        for (int m = 16; m <= 128;m*=2)
        {
            for (int efc = 50; efc <= 200; efc*=2)
            {
                recall_data_k(testind++, DIM, DATASIZE, m, efc, K, NUMQ, log, result, data, qs, gt);
                // recall_data_k(int testnum, int dim, int max_elements, int m, int ef_construction, int k, int numtest, 
                // std::ofstream& log, std::ofstream& result, float *data, float *qs, int *gt)
            }
        }
    }

    delete[] data;
    delete[] qs;
    delete[] gt;
    log.close();
    result.close();

    return 0;
}