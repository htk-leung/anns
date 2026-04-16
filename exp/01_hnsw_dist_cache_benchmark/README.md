## HNSW benchmarking with distance caching

Scripts written to test effects of distance caching using dataset SIFT. Tests also run on random data to verify that random data is harder than real-life datasets.

---
#### Distance caching infrastructure:

`Global dist_cache_ map (void*,void*)` → distance,count

`fstdistfunc_cache_()` distance calculation function incorporating caching enabled with macro `DISTCACHE`

`replace_fstdistfunc_for_cache()` function swaps HNSW distance function

`DISTCOUNT` measures exact distance function calls (construction+search)

`DISTCACHE` tests if caching reduces repeated computations