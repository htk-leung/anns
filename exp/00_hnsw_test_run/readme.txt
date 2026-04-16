


Aspect			hnswsearch_recall.cpp			hnswsearch_rand.cpp
Data source		Generated inside recall_rand_k(...)	Precomputed binary files
Datasets		Implicit / dynamic			Fixed dims 16, 32, 64
Search function		recall_rand_k(...)			recall_rand(...)
File output		20250926-hnsw.log, 20250926-hnsw.txt	20250926-hnsw-rand.log, 20250926-hnsw-rand.txt
Experiment style	“Generate new random data every call”	“Reuse stored random data”


-----------------------------------------------
__hnswsearch_recall.cpp

The first program is a benchmark driver that repeatedly calls recall_rand_k(...) for combinations of:

efs

M

ef_construction

dim

It does not load fixed datasets itself. Instead, it seems designed to generate fresh random data each time through recall_rand_k(...), as your comment says. Its purpose is to test many HNSW configurations on newly generated random inputs.


-----------------------------------------------
__hnswsearch_rand.cpp


What the second one does
The second program is a more explicit benchmark harness for three fixed random datasets at dimensions:

16

32

64

It loads pre-generated .bin files for data, queries, and ground truth, then calls recall_rand(...) for each parameter combination. So instead of generating a fresh random test inside the loop, it reuses stored random datasets


-----------------------------------------------
__hnswsearch_sift.cpp

This one is a fixed-dataset benchmark for SIFT, while the earlier recall_rand(...) version was for randomly generated datasets. It loads one large base set plus query and ground-truth files, then sweeps efs, M, and ef_construction over the same nested loops.

Main differences
This program uses a single dataset: SIFT with DIM = 128, DATASIZE = 1000000, NUMQ = 10000, and K = 100. The other version used three random datasets at dimensions 16, 32, and 64.

This one imports .fvecs and .ivecs files via xvec_to_arr(...), while the other version imports .bin files via bin_to_arr(...).

This program calls recall_data_k(...); the earlier one calls recall_rand(...). That suggests this file benchmarks a real dataset rather than synthetic random data.

Parameter sweep
The outer loops are the same idea as before: sweep efs from 10 to 80, M from 16 to 128, and ef_construction from 50 to 200. The key difference is that this version does not loop over dimension, because the dimension is fixed at 128.

I/O behavior
This program opens a single data file, query file, and ground-truth file, then closes them after loading arrays into memory. It also has a dedicated result log pair named 20250925-hnswsift.*, which separates it from the random-data benchmark.

Why it matters
Functionally, this version is closer to a standard HNSW recall benchmark on a known dataset, which is useful because recall depends heavily on M, efConstruction, and efSearch. Using a fixed benchmark dataset makes the results easier to compare across runs than generating new random data every time.