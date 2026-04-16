################################
# Begin ANN Benchmarks
# Ref: https://ann-benchmarks.com/
################################

all: deep-image-bench-flatnav deep-image-bench-hnsw

deep-image-bench-flatnav: 
	python3 run-benchmark.py \
		--dataset-name deep-image-96 \
		--dataset /T7/users/hl5737/data/deep1m/deep-image-96-angular.train.npy \
		--queries /T7/users/hl5737/data/deep1m/deep-image-96-angular.test.npy \
		--gtruth /T7/users/hl5737/data/deep1m/deep-image-96-angular.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
        --num-search-threads 1 \
		--metrics-file metrics.json

deep-image-bench-flatnavhnswbase: 
	python3 run-benchmark.py \
		--dataset-name deep-image-96 \
		--dataset /T7/users/hl5737/data/deep1m/deep-image-96-angular.train.npy \
		--queries /T7/users/hl5737/data/deep1m/deep-image-96-angular.test.npy \
		--gtruth /T7/users/hl5737/data/deep1m/deep-image-96-angular.gtruth.npy \
		--index-type flatnav \
		--hnsw-base-layer-filename deep-image.mtx \
		--use-hnsw-base-layer \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
        --num-search-threads 1 \
		--metrics-file metrics.json

deep-image-bench-hnsw: 
	python3 run-benchmark.py \
		--dataset-name deep-image-96 \
		--dataset /T7/users/hl5737/data/deep1m/deep-image-96-angular.train.npy \
		--queries /T7/users/hl5737/data/deep1m/deep-image-96-angular.test.npy \
		--gtruth /T7/users/hl5737/data/deep1m/deep-image-96-angular.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
        --num-search-threads 1 \
		--metrics-file metrics.json


# deep-image-bench-flatnav: 
# 	python3 run-benchmark.py \
# 		--dataset-name deep-image-96 \
# 		--dataset /root/data/deep1m/deep-image-96-angular.train.npy \
# 		--queries /root/data/deep1m/deep-image-96-angular.test.npy \
# 		--gtruth /root/data/deep1m/deep-image-96-angular.gtruth.npy \
# 		--index-type flatnav \
# 		--num-node-links 32 \
# 		--ef-construction 30 40 50 100 200 300 \
# 		--ef-search 100 200 300 500 1000 3000 \
# 		--metric l2 \
# 		--num-build-threads 16 \
#         --num-search-threads 1 \
# 		--metrics-file metrics-flatnav.json

# deep-image-bench-flatnavhnswbase: 
# 	python3 run-benchmark.py \
# 		--dataset-name deep-image-96 \
# 		--dataset /root/data/deep1m/deep-image-96-angular.train.npy \
# 		--queries /root/data/deep1m/deep-image-96-angular.test.npy \
# 		--gtruth /root/data/deep1m/deep-image-96-angular.gtruth.npy \
# 		--index-type flatnav \
# 		--hnsw-base-layer-filename deep-image.mtx \
# 		--use-hnsw-base-layer \
# 		--num-node-links 32 \
# 		--ef-construction 30 40 50 100 200 300 \
# 		--ef-search 100 200 300 500 1000 3000 \
# 		--metric l2 \
# 		--num-build-threads 16 \
#         --num-search-threads 1 \
# 		--metrics-file metrics.json

# deep-image-bench-hnsw: 
# 	python3 run-benchmark.py \
# 		--dataset-name deep-image-96 \
# 		--dataset /root/data/deep1m/deep-image-96-angular.train.npy \
# 		--queries /root/data/deep1m/deep-image-96-angular.test.npy \
# 		--gtruth /root/data/deep1m/deep-image-96-angular.gtruth.npy \
# 		--index-type hnsw \
# 		--num-node-links 32 \
# 		--ef-construction 30 40 50 100 200 300 \
# 		--ef-search 100 200 300 500 1000 3000 \
# 		--metric l2 \
# 		--num-build-threads 16 \
#         --num-search-threads 1 \
# 		--metrics-file metrics.json