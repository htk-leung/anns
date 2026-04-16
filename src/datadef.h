/*
    This header defines the datasets to run for hnsw and flatnav.
*/

#include <string>

// Define indices
#define DIM 0
#define DATASIZE 1
#define QNUM 2
#define GTNUM 3
#define M 4
#define EFC 5

#define NAME 0
#define SPACE 1

#define NUMTHREAD 1

/* -------------------------------------------------- */
/*
    Define metadata access params and
    corresponding dataset names and spaces, must align with actual data directory name
*/

#define DATASETS 5

int metadata[DATASETS][6] = {                   
    // dim, datasize, qnum, gtnum, m, efc, 
    { 96, 1000000, 1000, 100, 32, 100, },     // deep 
    // { 960, 1000000, 1000, 100, 32, 400, },    // gist m24efc100 low recall
    { 100, 1183514, 1000, 100, 40, 200, },    // glove100 m16efc100 bad performance
    { 200, 1183514, 1000, 100, 48, 300, },    // glove200
    // { 784, 60000, 10000, 100, 8, 400, },      // MNIST 784	                         <<< DATA PROBLEM?
    { 256, 290000, 10000, 100, 32, 200, },    // nytimes
    { 128, 1000000, 1000, 100, 16, 50, },     // sift
    // { 4, 1000000, 10000, 100, 7, 11, },       // rand1md4
    // { 8, 1000000, 10000, 100, 16, 50, },      // rand1md8
    // { 16, 1000000, 10000, 100, 16, 50, },     // rand1md16
    // { 32, 1000000, 10000, 100, 16, 50, },     // rand1md32
    // { 4, 10000, 100, 100, 4, 4, },            // randtest10k
};

std::string dataname[DATASETS][2] = {
    // name, space
    { "deep-image-96-angular", "Angular" },
    // { "gist-960-euclidean", "L2" },
    { "glove-100-angular", "Angular" },
    { "glove-200-angular", "Angular" },
    // { "mnist-784-euclidean", "L2" },
    { "nytimes-256-angular", "Angular" },
    { "sift-128-euclidean", "L2" },
    // { "rand1md4", "L2" },
    // { "rand1md8", "L2" },
    // { "rand1md16", "L2" },
    // { "rand1md32", "L2" },
    // { "randtest10k", "L2" }
};

/* -------------------------------------------------- */
/*
    Defines efs for m1-m4 double graph search
    each efs is pairs with the corresponding dist comp count from a normal hnsw run
*/ 

#define EFSNUM 13 // 13
int efsarr[] = { 10, 20, 50, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 3000 };

// m34
// from w13/results/data/hnsw-m5-<data>-opt.txt
// int dccount[DATASETS][EFSNUM] = {
//     { 188684 },  // deep
//     // {}, // gist
//     { 357678}, // glove100
//     { 367291}, // glove200
//     // {}, // mnist
//     { 427546}, // nytimes
//     { 151700 }, // sift
// };

// m567
// from w12/hnsw-n-<data>-opt.txt files
int dccount[DATASETS][EFSNUM] = {
    { 3212/2, 3212/2, 3212/2, 3212/2, 4679/2, 6143/2, 7607/2, 9065/2, 14849/2, 22034/2, 29159/2, 57360/2, 85256/2 },  // deep
    // {}, // gist
    { 4807/2, 4807/2, 4807/2, 4807/2, 6964/2, 9129/2, 11303/2, 13466/2, 22132/2, 32950/2, 43779/2, 86924/2, 129931/2 }, // glove100
    { 4952/2, 4952/2, 4952/2, 4952/2, 7108/2, 9280/2, 11443/2, 13600/2, 22232/2, 33000/2, 43731/2, 86475/2, 128937/2 }, // glove200
    // {}, // mnist
    { 3281/2, 3281/2, 3281/2, 3281/2, 4709/2, 6139/2, 7569/2, 8997/2, 14754/2, 21993/2, 29259/2, 58447/2, 87683/2 }, // nytimes
    { 2032/2, 2032/2, 2032/2, 2032/2, 2952/2, 3873/2, 4786/2, 5700/2, 9323/2, 13816/2, 18283/2, 35955/2, 53424/2 }, // sift
    // { 1300 }, // randtest10k 2579
};


/* -------------------------------------------------- */
/*
    Defines efc for m5-6 double graph search
*/ 

#define M56_MNUM 1
// std::string m56_mstr[] = { "8-", /*"16-", "32-", "64-",*/ };

/* -------------------------------------------------- */