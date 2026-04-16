#!/bin/bash 
# Usage: ./dl_datasets.sh <dataset_name> /path/to/dir/containing/data/ <normalize>

PYTHON=$(which python)

if [[ -z $PYTHON ]]; then 
    echo "Python not found. Please install python3."
    exit 1
fi

# Make sure we are in this directory before runnin

# Create a list of ANNS benchmark datasets to download.
ANN_BENCHMARK_DATASETS=("mnist-784-euclidean" 
                        "sift-128-euclidean" 
                        "glove-100-angular" 
                        "glove-200-angular" 
                        "deep-image-96-angular" 
                        "gist-960-euclidean" 
                        "nytimes-256-angular")

function print_help() {
    echo "Usage: ./download_anns_datasets.sh <dataset> [--normalize]"
    echo ""
    echo "Available datasets:"
    echo "${ANN_BENCHMARK_DATASETS[@]}"
    echo ""
    echo "Example Usage:"
    echo "  ./download_ann_benchmark_datasets.sh mnist-784-euclidean"
    echo "  ./download_ann_benchmark_datasets.sh glove-25-angular --normalize"
    exit 1
}


function download_dataset() {
    # Downloads a single benchmark dataset for Approximate Nearest Neighbor
    # Search (ANNS). Datasets are downloaded from http://ann-benchmarks.com/
    # and are stored in the data/ directory.

    local dataset=$1
    local dir=$2
    local normalize=$3

    echo "Downloading ${dataset}..."
    curl -L -o ${dataset}.hdf5 http://ann-benchmarks.com/${dataset}.hdf5

    # Create directory and move dataset to data/dataset_name.
    mkdir -p ${dir}data/${dataset}
    mv ${dataset}.hdf5 ${dir}data/${dataset}/${dataset}.hdf5

    # Create a set of training, query and groundtruth files by running the python 
    # script convert_ann_benchmark_datasets.py on the downloaded dataset. If normalize is set to 1, then pass 
    # the --normalize flag to convert_ann_benchmark_datasets.py.

    if [ ${normalize} -eq 1 ]; then
        echo "Converting ${dataset} to normalized npy..."
        $PYTHON convert_ann_benchmark_datasets.py ${dir}data/${dataset}/${dataset}.hdf5 --normalize
        echo "Converting ${dataset} to normalized xvecs..."
        $PYTHON conv_annbenchmark_to_xvecs.py ${dir}data/${dataset}/${dataset}.hdf5 --normalize
    else
        echo "Converting ${dataset} to npy..."
        $PYTHON convert_ann_benchmark_datasets.py ${dir}data/${dataset}/${dataset}.hdf5
        echo "Converting ${dataset} to xvecs..."
        $PYTHON conv_annbenchmark_to_xvecs.py ${dir}data/${dataset}/${dataset}.hdf5
    fi
}


# If the first argument is -h or --help, then print help and exit.
if [[ $1 == "-h" || $1 == "--help" ]]; then
    print_help
fi


# If ran with: ./download_anns_datasets.sh <dataset>, then 
# download only the specified dataset and do not normalize it.
if [[ $# -eq 1 ]]; then
    download_dataset $1 "" 0
    exit 0

elif [[ $# -eq 2 ]]; then
# If ran with: ./download_anns_datasets.sh <dataset> <dir> --normalize, then
# download only the specified dataset and normalize it.
    if [[ $2 == "--normalize" ]]; then
        download_dataset $1 "" 1
        exit 0
# If ran with: ./download_anns_datasets.sh <dataset> <dir>, then
# download only the specified dataset and save to designated directory.
    else
        download_dataset $1 $2 0
        exit 0
    fi
# If ran with: ./download_anns_datasets.sh <dataset> <dir> --normalize, then
# download only the specified dataset, save to designated directory and normalize it.
elif [[ $# -eq 3 ]]; then
    download_dataset $1 $2 1
    exit 0
fi
