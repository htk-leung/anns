#/bin/bash

PATH := /T7/users/hl5737/

mnist:
	./dl_datasets.sh mnist-784-euclidean ${PATH}

gist:
	./dl_datasets.sh gist-960-euclidean ${PATH}

deep:
	./dl_datasets.sh deep-image-96-angular ${PATH}

sift:
	./dl_datasets.sh sift-128-euclidean ${PATH}

nytimes:
	./dl_datasets.sh nytimes-256-angular ${PATH} --normalize

glove100:
	./dl_datasets.sh glove-100-angular ${PATH} --normalize

glove200:
	./dl_datasets.sh glove-200-angular ${PATH} --normalize