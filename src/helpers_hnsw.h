/*
    for running hnsw
*/
#include "../hnswlib/hnswlib/hnswlib.h"
#include <fstream>
#include <string>
#include <random>


float *find_queries(float *data, int dim, int numq, int max_elements);
int *find_gt(float *data, int k, int dim, int max_elements, float *q, int numq);
bool foundinapprox(std::pair<float, hnswlib::labeltype>q, std::priority_queue<std::pair<float, hnswlib::labeltype>>gt);
float recall_rand_k(int testnum, int dim, int max_elements, int m, int ef_construction, int k, int numtest, std::ofstream& log, std::ofstream& result);
float recall_rand_1(int testnum, int dim, int max_elements, int m, int ef_construction, std::ofstream& log);


// this one is the old version, but it really isn't necessary to do random sampling inside a randomly generated dataset
// new function in utils
float *find_queries(float *data, int dim, int numq, int datasize) 
{
    if(numq > datasize)
        throw std::runtime_error("find_queries : numq > data size");

    int size = numq * dim;
    float *qs = new float[size], *temp = qs; 

    for (int i = 0; i < size; i++) 
        qs[i] = data[i];
    
    return qs;
}

int *find_gt(float *data, int k, int dim, int max_elements, float *q, int numq) 
{
    #ifdef DEBUG
    std::cout<<"in func find_gt()"<<std::endl;
    #endif
    
    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::BruteforceSearch<float>* alg_brute = new hnswlib::BruteforceSearch<float>(&space, max_elements);

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        float* point_data = data + i * dim;
        alg_brute->addPoint(point_data, i);
    }

    // Query the elements for themselves
    std::priority_queue<std::pair<float, hnswlib::labeltype>> exact_result;
    int *gt = new int[k * numq]; // gt size of k results * numq queries
    int ind = 0;

    for (int i = 0; i < numq; i++) // for each query point
    {
        exact_result = alg_brute->searchKnn(q + dim*i, k); // searchKnn for each query at q + dim*i
        ind = k * (i+1) - 1;
        // std::cout << "i = " << i << " ";
        while(!exact_result.empty() && ind >= 0) {
            // std::cout << exact_result.top().second << " ";
            gt[ind--] = exact_result.top().second;
            exact_result.pop();
        }
        // std::cout << std::endl;
    }

    delete alg_brute;

    return gt;
}

// this returns recall for kNN
// find kNN for each and average out percentage
float recall_rand_k(int testnum, int dim, int max_elements, int m, int ef_construction, int k, int numtest, 
    std::ofstream& log, std::ofstream& result) 
{
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
    hnswlib::BruteforceSearch<float>* alg_brute = new hnswlib::BruteforceSearch<float>(&space, max_elements);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index | EDIT : add points to alg_brute
    for (int i = 0; i < max_elements; i++) {
        float* point_data = data + i * dim;
        alg_hnsw->addPoint(point_data, i);
        alg_brute->addPoint(point_data, i);
    }

    auto endtime = std::chrono::high_resolution_clock::now();
    auto duration = (endtime - starttime).count(); 
    log << "\tConstruction time\t: " << duration << "ms" << std::endl;
    starttime = std::chrono::high_resolution_clock::now();

    // Query the elements for themselves and measure recall | EDIT : change recall calculation
    // try to find the first numtest items
    float correct = 0;
    for (int i = 0; i < numtest; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> appro_result = alg_hnsw->searchKnn(data + i * dim, k);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> exact_result = alg_brute->searchKnn(data + i * dim, k);

        // EDIT : for each item in brute, find it in appro
        while(!exact_result.empty()) {
            if(foundinapprox(exact_result.top(), appro_result))
                correct++;
            exact_result.pop();
        }
    }

    endtime = std::chrono::high_resolution_clock::now();
    duration = (endtime - starttime).count(); 
    log << "\tSearch time\t\t\t: " << duration << "ms" << std::endl;

    float recall = correct / (numtest * k);
    std::cout << "kNN Test " << testnum << ":\nRecall: " << recall << std::endl;
    log << "\tRecall\t\t\t\t: " << recall << std::endl << std::endl;
    result << dim << "," << ef_construction << "," << m << k << "," << k << "," << recall << "," << duration << std::endl;     //"d,efc,m,efs,k,recall"

    delete alg_hnsw;
    delete alg_brute;
    return 0;
}

bool foundinapprox(std::pair<float, hnswlib::labeltype>q, std::priority_queue<std::pair<float, hnswlib::labeltype>>gt) 
{
    // binary search
    // compare label
    hnswlib::labeltype target = q.second;
    hnswlib::labeltype comp;
    while(!gt.empty()) 
    {
        if(target == gt.top().second) return true;
        gt.pop();
    }
    return false;
}

// this function is basically the copy of example_search.cpp
// recall = find each item and calculate percentage
float recall_rand_1(int testnum, int dim, int max_elements, int m, int ef_construction, std::ofstream& log) 
{
    // log
    log << "recall_efs_1 Test " << testnum << std::endl;
    log << "\tdim = " << dim
        << "\t max_elements = " << max_elements
        << "\tM = " << m
        << "\tef_construction = " << ef_construction << std::endl;
    auto starttime = std::chrono::high_resolution_clock::now();

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, m, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "1NN Test " << testnum << ":\nRecall: " << recall << "\n";

    // // Serialize index
    // std::string hnsw_path = "hnsw.bin";
    // alg_hnsw->saveIndex(hnsw_path);
    // delete alg_hnsw;

    // // Deserialize index and check recall
    // alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    // correct = 0;
    // for (int i = 0; i < max_elements; i++) {
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second; // this is why search > 1 gets recall = 0
    //     if (label == i) correct++;
    // }
    // recall = (float)correct / max_elements;
    // std::cout << "Recall of deserialized index: " << recall << "\n";

    auto endtime = std::chrono::high_resolution_clock::now();
    auto duration = (endtime - starttime).count(); 
    log << "\tElapsed time : " << duration << "ms" << std::endl;

    delete[] data;
    delete alg_hnsw;
    return 0;
}


float recall_data_k(int testnum, int dim, int max_elements, int m, int ef_construction, int k, int numq, 
    std::ofstream& log, std::ofstream& result, float *data, float *qs, int *gt) 
{
    log << "recall_data_k Test " << testnum << std::endl;
    log << "\t max_elements = " << max_elements
        << "\tdim = " << dim
        << "\tef_construction = " << ef_construction 
        << "\tM = " << m
        << "\tef_search = " << k
        << "\tk = " << k << std::endl;
    auto starttime = std::chrono::high_resolution_clock::now();

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, m, ef_construction);

    // Add data to index 
    for (int i = 0; i < max_elements; i++) 
    {
        float* point_data = data + i * dim;
        alg_hnsw->addPoint(point_data, i);
    }

    auto endtime = std::chrono::high_resolution_clock::now();
    auto duration = (endtime - starttime).count(); 
    log << "\tConstruction time\t: " << duration << "ms" << std::endl;
    starttime = std::chrono::high_resolution_clock::now();

    // Query & measure recall
    float correct = 0, val = 0;

    for (int i = 0; i < numq; i++) {                                    // for each query
        std::priority_queue<std::pair<float, hnswlib::labeltype>> appro_result = alg_hnsw->searchKnn(qs + i * dim, k);
        
        while(!appro_result.empty())                                    // for each item in appro, find it in gt
        { 
            val = appro_result.top().second;
            for (int j = 0; j < k; j++)
            {
                if (val == gt[i * k + j])
                {
                    correct++;
                    break;
                }
            }
            appro_result.pop();
        }
    }

    float recall = correct / (numq * k);

    endtime = std::chrono::high_resolution_clock::now();
    duration = (endtime - starttime).count(); 

    std::cout << "kNN Test " << testnum << ":\nRecall: " << recall << std::endl;

    log << "\tSearch time\t\t\t: " << duration << "ms" << std::endl;
    log << "\tRecall\t\t\t\t: " << recall << std::endl << std::endl;
    result << dim << "," << ef_construction << "," << m << k << "," << k << "," << recall << "," << duration << std::endl;     //"d,efc,m,efs,k,recall"

    delete alg_hnsw;
    return 0;
}

// next : separate indexing process from search

// int build_random_hnsw(int dim, int max_elements, int m, int ef_construction, char *filename) {
//     // Initing index | EDIT : alg_brute to find ground-truth kNN
//     hnswlib::L2Space space(dim);
//     hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

//     // Generate random data
//     std::mt19937 rng;
//     rng.seed(47);
//     std::uniform_real_distribution<> distrib_real;
//     float* data = new float[dim * max_elements];
//     for (int i = 0; i < dim * max_elements; i++) {
//         data[i] = distrib_real(rng);
//     }

//     // Add data to index | EDIT : add points to alg_brute
//     for (int i = 0; i < max_elements; i++) {
//         float* point_data = data + i * dim;
//         alg_hnsw->addPoint(point_data, i);
//     }

//     // Serialize index
//     std::string hnsw_path = filename;
//     alg_hnsw->saveIndex(hnsw_path);
//     delete alg_hnsw;

//     return 1;
// }