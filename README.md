# anns-research-hnsw
This repository is created to run tests on HNSW graphs.

Codes are written in C++, and HNSW library is sourced from 
https://github.com/nmslib/hnswlib.
https://github.com/BlaiseMuhirwa/flatnav

20250924        completed test runs of hnswlib with random library - code/hnswsearch_recall.cpp
                completed random data generation code - code/create_data_rand.cpp
                completed test runs of flatnav with random data generated ^^ file at - flatnav/tool/flatnav.cpp
                ** hnswlib is header-only, just need to include header files to run code
                ** flatnav is a library, need to build with ./bin/build.sh -e every time code changes.
