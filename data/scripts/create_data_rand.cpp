/*
    use this file to generate random fvecs/ivecs data files :

    * data                              float *
    * list of query points              float *
    * list of ground truth labels       int *
*/

/* ---EDIT THIS SECTION TO DETERMINE DATASET PROPERTIES--- */
#define DATAFILENAME "../../../../../T7/users/hl5737/data/rand1md32/data.fvecs"
#define QSFILENAME "../../../../../T7/users/hl5737/data/rand1md32/qs.fvecs"
#define GTFILENAME "../../../../../T7/users/hl5737/data/rand1md32/gt.ivecs"
#define DIM 32
#define N 1000000
#define K 100
#define NUMQUERIES 10000
/* ------------------------------------------------------- */

#include "../hnswlib/hnswlib/hnswlib.h"
#include "../src/utils.h"
#include "../code/helpers_hnsw.h"
#include <fstream>
#include <string>


int main() {
    std::cout << "Start program for " << DATAFILENAME << std::endl;

    // Create files
    std::ofstream datafile(DATAFILENAME, std::ios::binary | std::ios::out); 
    std::ofstream qsfile(QSFILENAME, std::ios::binary | std::ios::out); 
    std::ofstream gtfile(GTFILENAME, std::ios::binary | std::ios::out); 

    // Create data
    std::cout << "Creating data arr..." << std::endl;
    float *data = create_data(DIM, N);
    std::cout << "Creating qs arr..." << std::endl;
    float *qs = find_qs(data, DIM, NUMQUERIES);
    std::cout << "Creating gt arr..." << std::endl;
    int *gt = find_gt(data, K, DIM, N, qs, NUMQUERIES);
    
    #ifdef DEBUG
    std::cout<<"Finished creating arrays"<<std::endl;
    #endif
    // Save to file
    std::cout << "Creating data file..." << std::endl;
    if (!arr_to_xvecfile<float>(data, N, DIM, datafile, DATAFILENAME))
        throw std::runtime_error(std::string("Failed to write to file: ") + DATAFILENAME);
    std::cout << "Creating qs file..." << std::endl;
    if (!arr_to_xvecfile<float>(qs, NUMQUERIES, DIM, qsfile, QSFILENAME))
        throw std::runtime_error(std::string("Failed to write to file: ") + QSFILENAME);
    std::cout << "Creating gt file..." << std::endl;
    if (!arr_to_xvecfile<int>(gt, NUMQUERIES, K, gtfile, GTFILENAME))
        throw std::runtime_error(std::string("Failed to write to file: ") + GTFILENAME);

    // Close files
    gtfile.close();
    qsfile.close();
    datafile.close();

    // // Reopen gt file
    // std::ifstream readgt(GTFILENAME, std::ios::in | std::ios::binary);
    // int *readback = nullptr;
    // xvec_to_arr<int>(readgt, &readback, NUMQUERIES, K); // void xvec_to_arr(std::ifstream& file, datatype *outdata, int arrsize, int dim) { 

    // // Verify correctness, compare readback to original array
    // for (int i = 0; i < K * NUMQUERIES; i++)
    // {
    //     std::cout << "readback[" << i << "] = " << readback[i] << " gt[" << i << "] = " << gt[i] << std::endl;
    //     if (readback[i] != gt[i]) 
    //     {
    //         std::cout << "Results don't match!\n" << std::endl;
    //         delete[] data;
    //         delete[] qs;
    //         delete[] gt;
    //         return 1;
    //     }
    // }
    // readgt.close();
    // std::cout << "Results match!\n" << std::endl;

    // Clean up
    delete[] data;
    delete[] qs;
    delete[] gt;
    std::cout << "All done!" << std::endl;
    return 0;
}
