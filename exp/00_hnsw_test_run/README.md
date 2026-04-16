## HNSW test runs

Scripts written to build preliminary understanding of `hnswlib` and bit-level structure of datasets. Then advance from that to using standard benchmarking datasets.

---
#### hnswsearch_rand_gen.cpp

Benchmark script for testing many HNSW configurations on newly generated random inputs. It repeatedly calling `recall_rand_k()` for combinations of `efs`, `M`, `ef_construction`, and `dim`. Each call of `recall_rand_k(...)` generates a new dataset and uses `hnswlib::BruteforceSearch<float>` to compute ground truth to calculate recall.

<br>

---
#### hnswsearch_load_rand.cpp

Benchmarking script for testing HNSW search on pre-generated `.bin` files for data, queries, and ground truth. Datasets are generated with three fixed random datasets at dimensions `16`, `32`, and `64`. 

<br>

---
#### hnswsearch_load_sift.cpp

Benchmarking script for testing HNSW search on `SIFT`, a commonly used dataset that comes with data, queries, and ground truth sourced from `https://github.com/erikbern/ann-benchmarks`. It sweeps efs, M, and ef_construction over a range of values to test optimal configuration.

This version is the closest to a standard HNSW recall benchmark on a known dataset.