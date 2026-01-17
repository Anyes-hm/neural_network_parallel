/**
 * @file neural_network.c
 * @brief Hybrid MPI/OpenMP Neural Network Forward Pass
 * 
 * This implementation demonstrates parallel inference for a simple Multi-Layer Perceptron (MLP).
 * - MPI: Distributes input batches across processes
 * - OpenMP: Parallelizes matrix operations within each process
 * 
 * Architecture: Input(784) -> Hidden(128) -> Hidden(64) -> Output(10)
 * Dataset: MNIST-like (28x28 images, 10 classes)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

// Network architecture
#define INPUT_SIZE 784      // 28x28 images
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64
#define OUTPUT_SIZE 10      // 10 classes (digits 0-9)

// Training parameters
#define TOTAL_SAMPLES 1000  // Total number of samples
#define BATCH_SIZE 100      // Samples per MPI rank

/**
 * @struct NeuralNetwork
 * @brief Contains all weights and biases for the MLP
 */
typedef struct {
    // Layer 1: Input -> Hidden1
    float *w1;  // [INPUT_SIZE x HIDDEN1_SIZE]
    float *b1;  // [HIDDEN1_SIZE]
    
    // Layer 2: Hidden1 -> Hidden2
    float *w2;  // [HIDDEN1_SIZE x HIDDEN2_SIZE]
    float *b2;  // [HIDDEN2_SIZE]
    
    // Layer 3: Hidden2 -> Output
    float *w3;  // [HIDDEN2_SIZE x OUTPUT_SIZE]
    float *b3;  // [OUTPUT_SIZE]
} NeuralNetwork;

/**
 * @brief Initialize network with random weights (Xavier initialization)
 */
void init_network(NeuralNetwork *nn) {
    int i;
    
    // Allocate memory
    nn->w1 = (float*)malloc(INPUT_SIZE * HIDDEN1_SIZE * sizeof(float));
    nn->b1 = (float*)malloc(HIDDEN1_SIZE * sizeof(float));
    nn->w2 = (float*)malloc(HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float));
    nn->b2 = (float*)malloc(HIDDEN2_SIZE * sizeof(float));
    nn->w3 = (float*)malloc(HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->b3 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Xavier initialization for weights
    float std_w1 = sqrt(2.0 / (INPUT_SIZE + HIDDEN1_SIZE));
    float std_w2 = sqrt(2.0 / (HIDDEN1_SIZE + HIDDEN2_SIZE));
    float std_w3 = sqrt(2.0 / (HIDDEN2_SIZE + OUTPUT_SIZE));
    
    // Initialize weights with small random values
    for (i = 0; i < INPUT_SIZE * HIDDEN1_SIZE; i++) {
        nn->w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_w1;
    }
    for (i = 0; i < HIDDEN1_SIZE * HIDDEN2_SIZE; i++) {
        nn->w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_w2;
    }
    for (i = 0; i < HIDDEN2_SIZE * OUTPUT_SIZE; i++) {
        nn->w3[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_w3;
    }
    
    // Initialize biases to zero
    memset(nn->b1, 0, HIDDEN1_SIZE * sizeof(float));
    memset(nn->b2, 0, HIDDEN2_SIZE * sizeof(float));
    memset(nn->b3, 0, OUTPUT_SIZE * sizeof(float));
}

/**
 * @brief Free network memory
 */
void free_network(NeuralNetwork *nn) {
    free(nn->w1);
    free(nn->b1);
    free(nn->w2);
    free(nn->b2);
    free(nn->w3);
    free(nn->b3);
}

/**
 * @brief ReLU activation function
 */
static inline float relu(float x) {
    return x > 0 ? x : 0;
}

/**
 * @brief Softmax activation for output layer
 * @param input Input vector
 * @param output Output vector (probabilities)
 * @param size Vector size
 */
void softmax(float *input, float *output, int size) {
    float max_val = input[0];
    float sum = 0.0f;
    int i;
    
    // Find max for numerical stability
    for (i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    // Compute exp and sum
    for (i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

/**
 * @brief Matrix-vector multiplication with OpenMP parallelization
 * @param matrix Matrix [rows x cols]
 * @param vector Input vector [cols]
 * @param result Output vector [rows]
 * @param rows Number of rows
 * @param cols Number of columns
 */
void matmul_vec(float *matrix, float *vector, float *result, int rows, int cols) {
    int i, j;
    
    #pragma omp parallel for private(j) schedule(static)
    for (i = 0; i < rows; i++) {
        float sum = 0.0f;
        // Each thread computes one row
        for (j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}

/**
 * @brief Add bias and apply activation function
 * @param vector Input/output vector
 * @param bias Bias vector
 * @param size Vector size
 * @param activation 0=none, 1=ReLU
 */
void add_bias_activate(float *vector, float *bias, int size, int activation) {
    int i;
    
    #pragma omp parallel for
    for (i = 0; i < size; i++) {
        vector[i] += bias[i];
        if (activation == 1) {
            vector[i] = relu(vector[i]);
        }
    }
}

/**
 * @brief Forward pass for a single sample
 * @param nn Neural network
 * @param input Input vector [INPUT_SIZE]
 * @param output Output probabilities [OUTPUT_SIZE]
 */
void forward_pass(NeuralNetwork *nn, float *input, float *output) {
    // Temporary buffers for layer activations
    float hidden1[HIDDEN1_SIZE];
    float hidden2[HIDDEN2_SIZE];
    float logits[OUTPUT_SIZE];
    
    // Layer 1: Input -> Hidden1
    matmul_vec(nn->w1, input, hidden1, HIDDEN1_SIZE, INPUT_SIZE);
    add_bias_activate(hidden1, nn->b1, HIDDEN1_SIZE, 1); // ReLU
    
    // Layer 2: Hidden1 -> Hidden2
    matmul_vec(nn->w2, hidden1, hidden2, HIDDEN2_SIZE, HIDDEN1_SIZE);
    add_bias_activate(hidden2, nn->b2, HIDDEN2_SIZE, 1); // ReLU
    
    // Layer 3: Hidden2 -> Output
    matmul_vec(nn->w3, hidden2, logits, OUTPUT_SIZE, HIDDEN2_SIZE);
    add_bias_activate(logits, nn->b3, OUTPUT_SIZE, 0); // No activation
    
    // Softmax for probabilities
    softmax(logits, output, OUTPUT_SIZE);
}

/**
 * @brief Generate synthetic input data (simulating MNIST)
 */
void generate_synthetic_data(float *data, int num_samples) {
    int i, j;
    
    for (i = 0; i < num_samples; i++) {
        for (j = 0; j < INPUT_SIZE; j++) {
            // Random normalized pixel values [0, 1]
            data[i * INPUT_SIZE + j] = (float)rand() / RAND_MAX;
        }
    }
}

/**
 * @brief Main function: Distributed inference with MPI + OpenMP
 */
int main(int argc, char **argv) {
    int rank, size;
    double start_time, end_time, compute_time;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set random seed (different for each rank)
    srand(time(NULL) + rank);
    
    // Set OpenMP threads
    int num_threads = omp_get_max_threads();
    if (rank == 0) {
        printf("====================================\n");
        printf("Parallel Neural Network Forward Pass\n");
        printf("====================================\n");
        printf("MPI Processes: %d\n", size);
        printf("OpenMP Threads per process: %d\n", num_threads);
        printf("Total Samples: %d\n", TOTAL_SAMPLES);
        printf("Samples per rank: %d\n", BATCH_SIZE);
        printf("Network: %d -> %d -> %d -> %d\n", 
               INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
        printf("====================================\n\n");
    }
    
    // Allocate local data for this rank
    float *local_inputs = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    float *local_outputs = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    
    // Initialize network (same weights on all ranks)
    NeuralNetwork nn;
    init_network(&nn);
    
    // Generate synthetic input data
    generate_synthetic_data(local_inputs, BATCH_SIZE);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // ========================================
    // PARALLEL FORWARD PASS
    // Each rank processes its batch of samples
    // ========================================
    int i;
    for (i = 0; i < BATCH_SIZE; i++) {
        forward_pass(&nn, 
                    &local_inputs[i * INPUT_SIZE], 
                    &local_outputs[i * OUTPUT_SIZE]);
    }
    
    // Synchronize after computation
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    compute_time = end_time - start_time;
    
    // ========================================
    // GATHER RESULTS TO RANK 0
    // ========================================
    float *all_outputs = NULL;
    if (rank == 0) {
        all_outputs = (float*)malloc(TOTAL_SAMPLES * OUTPUT_SIZE * sizeof(float));
    }
    
    MPI_Gather(local_outputs, BATCH_SIZE * OUTPUT_SIZE, MPI_FLOAT,
               all_outputs, BATCH_SIZE * OUTPUT_SIZE, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    
    // ========================================
    // PERFORMANCE REPORTING
    // ========================================
    double max_time;
    MPI_Reduce(&compute_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Performance Results:\n");
        printf("--------------------\n");
        printf("Total inference time: %.6f seconds\n", max_time);
        printf("Throughput: %.2f samples/second\n", TOTAL_SAMPLES / max_time);
        printf("Time per sample: %.6f seconds\n\n", max_time / BATCH_SIZE);
        
        // Show sample predictions
        printf("Sample Predictions (first 3 samples):\n");
        printf("-------------------------------------\n");
        for (i = 0; i < 3; i++) {
            printf("Sample %d: [", i);
            int j;
            for (j = 0; j < OUTPUT_SIZE; j++) {
                printf("%.3f", all_outputs[i * OUTPUT_SIZE + j]);
                if (j < OUTPUT_SIZE - 1) printf(", ");
            }
            printf("]\n");
            
            // Find predicted class
            int max_idx = 0;
            float max_prob = all_outputs[i * OUTPUT_SIZE];
            for (j = 1; j < OUTPUT_SIZE; j++) {
                if (all_outputs[i * OUTPUT_SIZE + j] > max_prob) {
                    max_prob = all_outputs[i * OUTPUT_SIZE + j];
                    max_idx = j;
                }
            }
            printf("          Predicted class: %d (confidence: %.2f%%)\n\n", 
                   max_idx, max_prob * 100);
        }
        
        free(all_outputs);
    }
    
    // Cleanup
    free(local_inputs);
    free(local_outputs);
    free_network(&nn);
    
    MPI_Finalize();
    return 0;
}
