## HNSW multi-graph search approach

Experiments conducted based on assumption that different graph structure leads to different nodes to be found with the same BEAM search parameters. One approach is to combine results from multiple graphs. 

Odd-numbered models do not randomize data-point insertion sequence between graphs, evenly-numbered ones do. 

<br>

---
#### Models 3-4 : buggy basic setup

Basic setup

<br>

---
#### Model 5

- clean-up and optimization from M3
- no distance caching between graphs
- dcbudget defined from normal hnsw search run
- build with M/2

<br>

---

#### Models 7-8 :

- New class `hnswM<float>` is defined and saved in `hnswlib` to support :
    - cached-distance sharing between graphs
    - cache-miss metric
    - % visit-overlap between 2 graphs
    - search results merging

- differing distance computation budget split between models:
    - `M7` 1:1
    - `M8` 1.5:0.5