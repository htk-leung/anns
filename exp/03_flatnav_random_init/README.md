## FlatNav benchmarking with custom graph initialization parameters

FlatNav proposes that the benefits of the hierarchical structure of HNSW is minimal for search performance. It replaces the hierarchy with a randomized graph initialization technique to find the closest point at the based layer of the graph to begin BEAM search. 

The script tests varying parameter input into the initialization function to understand its significance and impact.