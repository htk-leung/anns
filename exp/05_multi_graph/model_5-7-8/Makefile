SRCPATH := ../../src/
HNSWPATH := ../../hnswlib/hnswlib/
DEPS := ${wildcard *.h}
SRCDEPS := ${wildcard ${SRCPATH}*.h}
HNSWDEPS := ${wildcard ${HNSWPATH}*.h}

CXX := g++
CXXFLAGS := -Ofast -funroll-loops

# in general
# add -DFILEAPP if appending to existing file
######################
# W12
######################
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
	./$@

# 2 graphs, change layer assignment
hnswm: hnsw_mul.cpp ${DEPS}
	${CXX} hnsw_mul.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./$@

# 2 graphs, change layer assignment + change insertion order
hnswm2: hnsw_mul2.cpp ${DEPS}
	${CXX} hnsw_mul2.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./$@

# 2 graphs, change layer assignment 
hnswm3: hnsw_mul3.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul3.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./$@
hnswm3debug: hnsw_mul3.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul3.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -o $@
	./$@
hnswm3debuga: hnsw_mul3.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul3.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -DFILEAPP -o $@
	./$@

# 2 graphs, change layer assignment + change insertion order
hnswm4: hnsw_mul4.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul4.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./$@

######################
# W13
######################

# 2 graphs, change layer assignment + budget / 2, changing M to compare skinniness
hnswm5: hnsw_mul5.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul5.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./$@
hnswm5debug: hnsw_mul5.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul5.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -o $@
	./$@
hnswm5debuga: hnsw_mul5.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul5.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -DFILEAPP -o $@
	./$@

# 2 graphs, change layer assignment + budget / 2, changing M to compare skinniness
hnswm6: hnsw_mul6.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul6.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./$@
hnswm6debug: hnsw_mul6.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul6.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -o $@
	./$@
hnswm6debuga: hnsw_mul6.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul6.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -DFILEAPP -o $@
	./$@

hnswm7: hnsw_mul7.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul7.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./$@
hnswm7debug: hnsw_mul7.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul7.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -o $@
	./$@
hnswm7debuga: hnsw_mul7.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul7.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -DDEBUG -DFILEAPP -o $@
	./$@

hnswm8: hnsw_mul8.cpp ${SRCDEPS} ${HNSWDEPS}
	${CXX} hnsw_mul8.cpp -I${HNSWPATH} -I${SRCPATH} ${CXXFLAGS} -DOPT -o $@
	./$@

clean:
	rm -f hnswopt hnswoptdebug hnswoptdebuga hnswnorm \
	hnswm \
	hnswm2 \
	hnswm3 hnswm3debug hnswm3debuga \
	hnswm4 \
	hnswm5 hnswm5debug hnswm5debuga \
	hnswm6 hnswm6debug hnswm6debuga \
	hnswm7 hnswm7debug hnswm7debuga \
	hnswm8