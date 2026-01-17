#!/bin/bash

# Installation Script for Neural Network MPI/OpenMP Project
# Supports Ubuntu/Debian, macOS (Homebrew), and WSL2

set -e  # Exit on error

echo "=========================================="
echo "Neural Network Parallel Project - Setup"
echo "=========================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if grep -q Microsoft /proc/version 2>/dev/null; then
        OS="wsl"
        echo "✓ Detected: WSL (Windows Subsystem for Linux)"
    else
        echo "✓ Detected: Linux"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "✓ Detected: macOS"
else
    echo "✗ Unsupported OS: $OSTYPE"
    exit 1
fi

echo ""

# Install dependencies based on OS
case $OS in
    linux|wsl)
        echo "Installing dependencies via apt..."
        
        # Check if running with sudo
        if [ "$EUID" -ne 0 ]; then 
            echo "⚠ This script needs sudo privileges to install packages."
            echo "Please run: sudo $0"
            exit 1
        fi
        
        # Update package list
        echo "Updating package list..."
        apt update -qq
        
        # Install build tools
        echo "Installing build-essential..."
        apt install -y build-essential cmake > /dev/null 2>&1
        
        # Install MPI
        echo "Installing OpenMPI..."
        apt install -y openmpi-bin openmpi-common libopenmpi-dev > /dev/null 2>&1
        
        # Install OpenMP (usually included with gcc)
        echo "Installing OpenMP support..."
        apt install -y libomp-dev > /dev/null 2>&1
        
        # Install Python3 (for benchmarking)
        echo "Installing Python3..."
        apt install -y python3 > /dev/null 2>&1
        
        echo "✓ All dependencies installed!"
        ;;
        
    macos)
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "✗ Homebrew not found. Please install it first:"
            echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        
        echo "Installing dependencies via Homebrew..."
        
        # Install GCC (includes OpenMP)
        echo "Installing GCC..."
        brew install gcc > /dev/null 2>&1 || true
        
        # Install OpenMPI
        echo "Installing OpenMPI..."
        brew install open-mpi > /dev/null 2>&1 || true
        
        # Install Python3
        echo "Installing Python3..."
        brew install python3 > /dev/null 2>&1 || true
        
        echo "✓ All dependencies installed!"
        echo ""
        echo "⚠ Note: On macOS, you may need to use 'mpicc' from Homebrew."
        echo "  Make sure /usr/local/bin is in your PATH."
        ;;
esac

echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

# Verify GCC
echo -n "Checking GCC... "
if command -v gcc &> /dev/null; then
    echo "✓ $(gcc --version | head -n 1)"
else
    echo "✗ Not found"
    exit 1
fi

# Verify MPI
echo -n "Checking MPI... "
if command -v mpicc &> /dev/null; then
    echo "✓ $(mpicc --version | head -n 1)"
else
    echo "✗ Not found"
    exit 1
fi

# Verify mpirun
echo -n "Checking mpirun... "
if command -v mpirun &> /dev/null; then
    echo "✓ $(mpirun --version 2>&1 | head -n 1)"
else
    echo "✗ Not found"
    exit 1
fi

# Verify OpenMP support
echo -n "Checking OpenMP... "
if echo | gcc -fopenmp -dM -E - 2>/dev/null | grep -q _OPENMP; then
    echo "✓ Supported"
else
    echo "✗ Not supported"
    exit 1
fi

# Verify Python3
echo -n "Checking Python3... "
if command -v python3 &> /dev/null; then
    echo "✓ $(python3 --version)"
else
    echo "⚠ Not found (optional, needed for benchmark.py)"
fi

echo ""
echo "=========================================="
echo "Testing Compilation"
echo "=========================================="

# Try to compile a simple MPI+OpenMP program
echo "Creating test program..."
cat > /tmp/test_mpi_omp.c << 'EOF'
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    #pragma omp parallel
    {
        #pragma omp master
        if (rank == 0) {
            printf("MPI + OpenMP test: SUCCESS\n");
            printf("MPI processes: 1, OpenMP threads: %d\n", omp_get_num_threads());
        }
    }
    
    MPI_Finalize();
    return 0;
}
EOF

echo -n "Compiling test program... "
if mpicc -fopenmp /tmp/test_mpi_omp.c -o /tmp/test_mpi_omp 2>/dev/null; then
    echo "✓"
    
    echo -n "Running test program... "
    if mpirun -np 1 /tmp/test_mpi_omp > /tmp/test_output.txt 2>&1; then
        echo "✓"
        cat /tmp/test_output.txt
    else
        echo "✗ Failed to run"
        cat /tmp/test_output.txt
        exit 1
    fi
    
    # Cleanup
    rm -f /tmp/test_mpi_omp /tmp/test_mpi_omp.c /tmp/test_output.txt
else
    echo "✗ Failed to compile"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You can now compile and run the project:"
echo "  cd neural_network_parallel"
echo "  make"
echo "  make run"
echo ""
echo "For benchmarking:"
echo "  make benchmark"
echo "  # or"
echo "  python3 benchmark.py"
echo ""
echo "For help:"
echo "  make help"
echo "  cat README.md"
echo ""
echo "=========================================="
