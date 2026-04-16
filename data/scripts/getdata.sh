# Setup
# Run separately in Acaconda Prompt : conda install anaconda::h5py
# assume running from anns-research-hnsw directory
mkdir data
cd data

# sift Euclidean (1000000 x 128)
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzvf sift.tar.gz
rm sift.tar.gz

# gist Euclidean (1000000 x 960)
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzvf gist.tar.gz
rm gist.tar.gz

# deep1m (1000000 x 256) 
wget https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz
tar -xzvf deep1M.tar.gz
rm deep1M.tar.gz

# GloVe100 angular (1183514 x 100)
# python with Conda
wget http://ann-benchmarks.com/glove-100-angular.hdf5
python python convert_ann_benchmark_datasets.py glove100/glove100.hdf5 --normalize 

# GloVe200 angular (1183514 x 200)
# python with Conda
wget http://ann-benchmarks.com/glove-100-angular.hdf5
python python convert_ann_benchmark_datasets.py glove200/glove200.hdf5 --normalize