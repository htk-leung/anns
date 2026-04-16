SRCPATH := ../../src
HNSWPATH := ../../hnswlib/hnswlib
DEPS := ${wildcard *.h}
FLAGS := -Ofast -funroll-loops

hnsw: hnsw.cpp ${DEPS}
	g++ hnsw.cpp -I${HNSWPATH} -I${SRCPATH} ${FLAGS} -o hnsw -DFILEAPP