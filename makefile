# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++14 -Wall -O2 -g -I.

# Special flags
OMPFLAGS = -fopenmp

# Object files
OBJS = nmf.o nmf_sparse.o nnls.o predict.o readInput.o bits.o

# Default target
all: nmf

# Compile rules
nmf.o: nmf.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

nmf_sparse.o: nmf_sparse.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

nnls.o: nnls.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

readInput.o: readInput.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

predict.o: predict.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

bits.o: bits.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

# Link everything
nmf: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(OBJS) -o nmf

# Clean up
clean:
	rm -f *.o nmf