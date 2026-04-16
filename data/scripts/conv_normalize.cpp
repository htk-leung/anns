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
            xvecfile_to_arr<float>(path, dataname[i][NAME], "data.fvecs", &data, metadata[i][DATASIZE], metadata[i][DIM]);
            xvecfile_to_arr<float>(path, dataname[i][NAME], "qs.fvecs", &qs, metadata[i][QNUM], metadata[i][DIM]); 

            if (data == nullptr || qs == nullptr) {
                throw std::runtime_error("Pointers are still null after allocation");
            }
        } catch (const std::exception& e) {
            std::cout << "Error : " << e.what() << std::endl;

            delete[] data;
            delete[] qs;
            return 1;
        };

        // Create result file
        std::string datapath = path + dataname[i][NAME] + "/" + "datanorm.fvecs";
        std::ofstream datafile(datapath, std::ios::binary | std::ios::out); 
        std::string qspath = path + dataname[i][NAME] + "/" + "qsnorm.fvecs";
        std::ofstream qsfile(qspath, std::ios::binary | std::ios::out); 

        // Print values before normalization
        std::cout << "data :";
        for (int i = 0; i < 10; i++)
            std::cout << " " << data[i] << (i == 9 ? "\n" : ""); 
        std::cout << "qs :";
        for (int i = 0; i < 10; i++)
            std::cout << " " << qs[i] << (i == 9 ? "\n" : ""); 

        // Normalize
        normalize_vectors(data, metadata[i][DATASIZE], metadata[i][DIM]);
        normalize_vectors(qs, metadata[i][QNUM], metadata[i][DIM]);

        // Print values after normalization
        std::cout << "data :";
        for (int i = 0; i < 10; i++)
            std::cout << " " << data[i] << (i == 9 ? "\n" : ""); 
        std::cout << "qs :";
        for (int i = 0; i < 10; i++)
            std::cout << " " << qs[i] << (i == 9 ? "\n" : ""); 

        // Write to file
        arr_to_xvecfile<float>(data, metadata[i][DATASIZE], metadata[i][DIM], datafile, datapath);
        arr_to_xvecfile<float>(qs, metadata[i][QNUM], metadata[i][DIM], qsfile, qspath);
        
        // Save results
        datafile.close();
        qsfile.close();

        // Clean up
        delete[] data;
        delete[] qs;
    }
}