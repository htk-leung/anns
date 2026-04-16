# anns-research-hnsw
Approximate nearest neighbor search algorithm optimization research project focusing on `HNSW` (Hierarchical Navigable Small World) and `FlatNav`.

<br>

---
#### Problem

ANN (Approximate Nearest Neighbor) search is a data-indexing technique used to find, with high probability, data point(s) in a large dataset that is very close to a query point, prioritizing search speed over absolute precision. 

Exact kNN algorithms incur `O(n)` runtime, which is theoretically acceptable but becomes costly as datasets grow large (100K to 1M+ high-dimensional vectors). 

Since data retrieval is foundational for data-intensive technologies, including AI/ML, search engines, computer vision, and natural language processing, performance optimization is critical. ANN algorithms offer relief by relaxing the search criteria from "exact" to "approximate", turning the problem into one of optimizing `recall` (search accuracy) while minimizing `latency` (time cost).

<br>

---
#### Research 

The project began with test runs of `HNSW` to understand the library's source code and replicate benchmarking results. The discovery of the `FlatNav` paper sparked a period of experimentation with graph initialization techniques, exploring the possibilities and limits of random initialization without the hierarchical structure of `HNSW`. 

Further experiments on distance caching and multiple-index search methods gave insight into search behavior under varying graph quality. Current efforts focus on HNSW initialization, with coarse-to-fine-grained search methods planned for future experimentation.

For details, refer to `exp`.

<br>

---
#### Repository structure

`lib` Libraries used in this project. Modified to run experiments.

`exp` Experiments done.

`src` Shared header files between experiments.

`data` Scripts used to download, convert, generate data used.