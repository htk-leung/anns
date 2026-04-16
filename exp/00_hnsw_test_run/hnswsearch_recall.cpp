/*
    This code calls the function

        recall_rand_k(...)

    to generate a new set of random data for a test each time it's called.

    WORKS DON'T TOUCH
*/

#include "../hnswlib/hnswlib/hnswlib.h"
#include "helpers.h"
#include "helpers_hnsw.h"

#include <fstream>
#include <chrono>

#define logfile "../logresult/20250926-hnsw.log"
#define resultfile "../logresult/20250926-hnsw.txt"

int main() {
    // create log in append mode
    std::ofstream log(logfile, std::ios::out); 
    std::ofstream result(resultfile, std::ios::out); 

    result << "d,efc,m,efs,k,recall,searchtime" << std::endl;

    // float recall_efs_1(testnum, 16, 10000, 16, 200, 10, log); // original config

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
                    recall_rand_k(testind++, d, 10000, m, efc, efs, 100, log, result);
                }
            }
        }
    }

    return 0;
}