#include "../src/utils.h"
#include "../src/datadef.h"
#include <string>
#include <fstream>
#include <iostream>

int main() {
    // for each dataset
    std::string path = "../../../../T7/users/hl5737/data/";
    for (int i = 0; i < DATASETS; i++)
    {
        // Array variables
        float *data = nullptr, *qs = nullptr;

        // Read xvec file to arr
        try {
            xvecfile_to_arr<float>(path, dataname[i][NAME], "data.fvecs", &data, 1, metadata[i][DIM]);
            xvecfile_to_arr<float>(path, dataname[i][NAME], "qs.fvecs", &qs, 1, metadata[i][DIM]); 

            if (data == nullptr || qs == nullptr) {
                throw std::runtime_error("Pointers are still null after allocation");
            }
        } catch (const std::exception& e) {
            std::cout << "Error : " << e.what() << std::endl;

            delete[] data;
            delete[] qs;
            return 1;
        };

        // Print first 10 values 
        std::cout << dataname[i][NAME] << std::endl;
        int n = 50;
        std::cout << "\tdata :";
        for (int i = 0; i < n; i++)
            std::cout << " " << data[i] << (i == n-1 ? "\n" : ""); 
        std::cout << "\tqs :";
        for (int i = 0; i < n; i++)
            std::cout << " " << qs[i] << (i == n-1 ? "\n" : ""); 

        // Clean up
        delete[] data;
        delete[] qs;
    }
}