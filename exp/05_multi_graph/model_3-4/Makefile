SRCPATH := ../../src/
HNSWPATH := ../../hnswlib/hnswlib/
DEPS := ${wildcard *.h}
SRCDEPS := ${wildcard ${SRCPATH}*.h}
HNSWDEPS := ${wildcard ${HNSWPATH}*.h}

CXX := g++
CXXFLAGS := -Ofast -funroll-loops

# in general
# add -DFILEAPP if appending to existing file

hnswopt: hnsw.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o hnsw 
	./hnsw

hnswoptdebug: hnsw.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -o $@
	./$@

hnswoptdebuga: hnsw.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -DFILEAPP -o $@
	./$@

hnswnorm: hnsw.cpp ${DEPS}
	${CXX} hnsw.cpp -o hnswn
	./hnswn

# 2 graphs, change layer assignment
hnswm: hnsw_mul.cpp ${DEPS}
	${CXX} hnsw_mul.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./hnswm

# 2 graphs, change layer assignment + insertion order
hnswm2: hnsw_mul2.cpp ${DEPS}
	${CXX} hnsw_mul2.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./hnswm2

# 2 graphs, change layer assignment + insertion order + budget / 2
hnswm3: hnsw_mul3.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul3.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./hnswm3

hnswm3debug: hnsw_mul3.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul3.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -o $@
	./$@

hnswm3debuga: hnsw_mul3.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul3.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -DFILEAPP -o $@
	./$@

# 2 graphs, change layer assignment + budget / 2
hnswm4: hnsw_mul4.cpp ${DEPS}
	${CXX} hnsw_mul4.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./hnswm4


clean:
	rm -f hnswopt hnswoptdebug hnswnorm hnswm hnswm2 hnswm3 hnswm4