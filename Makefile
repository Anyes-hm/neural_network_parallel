# Makefile for Hybrid MPI/OpenMP Neural Network
# Author: Master 2 IA Project
# Description: Compiles and runs parallel neural network inference

# Compiler settings
MPICC = mpicc
CFLAGS = -O3 -fopenmp -Wall -Wextra -march=native
LDFLAGS = -lm -fopenmp

# Target executables
TARGET = neural_network
TARGET_SERIAL = neural_network_serial
SOURCE = neural_network.c
SOURCE_SERIAL = neural_network_serial.c

# Number of MPI processes (adjust based on your system)
NP = 4

# Default target
all: $(TARGET) $(TARGET_SERIAL)

# Compile the parallel program
$(TARGET): $(SOURCE)
	$(MPICC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)
	@echo "✓ Parallel version compiled!"

# Compile the serial baseline
$(TARGET_SERIAL): $(SOURCE_SERIAL)
	gcc -O3 -Wall -o $(TARGET_SERIAL) $(SOURCE_SERIAL) -lm
	@echo "✓ Serial baseline compiled!"

# Run with default settings
run: $(TARGET)
	@echo "Running with $(NP) MPI processes..."
	mpirun -np $(NP) ./$(TARGET)

# Run with different numbers of processes for benchmarking
benchmark: $(TARGET)
	@echo "=== Benchmarking with different MPI configurations ==="
	@echo ""
	@echo ">>> 1 process (serial baseline):"
	@OMP_NUM_THREADS=1 mpirun -np 1 ./$(TARGET)
	@echo ""
	@echo ">>> 2 processes, 2 threads each:"
	@OMP_NUM_THREADS=2 mpirun -np 2 ./$(TARGET)
	@echo ""
	@echo ">>> 4 processes, 2 threads each:"
	@OMP_NUM_THREADS=2 mpirun -np 4 ./$(TARGET)
	@echo ""
	@echo ">>> 4 processes, 4 threads each:"
	@OMP_NUM_THREADS=4 mpirun -np 4 ./$(TARGET)

# Test correctness with small dataset
test: $(TARGET)
	@echo "Running correctness test..."
	OMP_NUM_THREADS=1 mpirun -np 1 ./$(TARGET)

# Compare serial vs parallel
compare: $(TARGET) $(TARGET_SERIAL)
	@echo "=== Comparing Serial vs Parallel Performance ==="
	@echo ""
	@echo ">>> Serial baseline:"
	@./$(TARGET_SERIAL)
	@echo ""
	@echo ">>> Parallel (4 processes, 2 threads):"
	@OMP_NUM_THREADS=2 mpirun -np 4 ./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TARGET_SERIAL) *.o

# Show compilation info
info:
	@echo "Compiler: $(MPICC)"
	@echo "Flags: $(CFLAGS)"
	@echo "OpenMP support: $(shell $(MPICC) -fopenmp -dM -E - < /dev/null | grep -c _OPENMP)"
	@echo "MPI version: $(shell mpirun --version | head -n 1)"

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Compile both parallel and serial versions"
	@echo "  make run      - Run parallel version with 4 MPI processes"
	@echo "  make benchmark- Run performance tests"
	@echo "  make compare  - Compare serial vs parallel performance"
	@echo "  make test     - Test correctness"
	@echo "  make clean    - Remove compiled files"
	@echo "  make info     - Show compiler information"

.PHONY: all run benchmark test clean info help
