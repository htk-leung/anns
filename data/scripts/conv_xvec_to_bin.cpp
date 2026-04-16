/*
    use this file to select 10000 data items, 
    100 guery points and 100 ground truths for each query point 
    from fvec datatsets.

    *** NOTE that because the original dataset size is 1m
    downsizing will render MANY query and groundtruth points out of range
    therefore they MUST be reconstructed from scratch
*/

/* EDIT THIS SECTION TO DETERMINE DATASET PROPERTIES*/
#define IDATAFILENAME "../data/sift1m/sift_base.fvecs"

#define ODATAFILENAME "../data/sift1m/sift10k_base.bin"
#define OQSFILENAME "../data/sift1m/sift10k_query100.bin"
#define OGTFILENAME "../data/sift1m/sift10k_groundtruth100.bin"

#define DIM 128
#define DATASIZE 10000
#define QNUM 100
#define K 100

#include "../hnswlib/hnswlib/hnswlib.h"
#include "helpers.h"
#include "helpers_hnsw.h"
#include <fstream>
#include <string>


int main() {
    // Create files in append mode
    std::ifstream idatafile(IDATAFILENAME, std::ios::binary); 

    std::ofstream odatafile(ODATAFILENAME, std::ios::binary | std::ios::out); 
    std::ofstream oqsfile(OQSFILENAME, std::ios::binary | std::ios::out); 
    std::ofstream ogtfile(OGTFILENAME, std::ios::binary | std::ios::out); 

    // Read data
    float *data=nullptr;

    xvec_to_arr<float>(idatafile, &data, DATASIZE, DIM); 
    idatafile.close();

    #ifdef DEBUG
        std::cout << "Data validation:" << std::endl;
        std::cout << "First data point: ";
        for (int i = 0; i < 5; i++) std::cout << data[i] << " ";
        std::cout << std::endl;
    #endif

    float *qs = find_queries(data, DIM, QNUM, DATASIZE);
    int *gt = find_gt(data, K, DIM, DATASIZE, qs, QNUM);
    
    #ifdef DEBUG
        std::cout<<"finished creating arrays"<<std::endl;

        std::cout << "Data validation:" << std::endl;

        std::cout << "First query point: ";
        for (int i = 0; i < 5; i++) std::cout << qs[i] << " ";
        std::cout << std::endl;

        std::cout << "First groundtruth point: ";
        for (int i = 0; i < 5; i++) std::cout << gt[i] << " ";
        std::cout << std::endl;
    #endif

    // Save to file
    if (!arr_to_bin<float>(data, DATASIZE, DIM, odatafile, ODATAFILENAME))
        throw std::runtime_error(std::string("Failed to write to file: ") + ODATAFILENAME);
    if (!arr_to_bin<float>(qs, QNUM, DIM, oqsfile, OQSFILENAME))
        throw std::runtime_error(std::string("Failed to write to file: ") + OQSFILENAME);
    if (!arr_to_bin<int>(gt, QNUM, K, ogtfile, OGTFILENAME))
        throw std::runtime_error(std::string("Failed to write to file: ") + OGTFILENAME);

    #ifdef DEBUG
    std::cout<<"finished saving to file"<<std::endl;
    #endif

    // Close files
    ogtfile.close();
    oqsfile.close();
    odatafile.close();

    // Reopen gt file
    std::ifstream readgt(OGTFILENAME, std::ios::in | std::ios::binary);
    int *readback = bin_to_arr<int>(readgt, OGTFILENAME, K);

    // Verify correctness, compare readback to original array
    for (int i = 0; i < K * QNUM; i++)
    {
        if (readback[i] != gt[i]) 
        {
            std::cout << "Results don't match!\n" << std::endl;
            delete[] data;
            delete[] qs;
            delete[] gt;
            return 1;
        }
    }

    std::cout << "Results match!\n" << std::endl;

    // Clean up
    delete[] data;
    delete[] qs;
    delete[] gt;
    delete[] readback;
    return 0;
}
