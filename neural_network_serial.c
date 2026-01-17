/**
 * @file neural_network_serial.c
 * @brief Sequential Neural Network Forward Pass (Baseline for comparison)
 * 
 * This is the sequential baseline version without MPI or OpenMP.
 * Used to measure speedup of the parallel version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64
#define OUTPUT_SIZE 10
#define TOTAL_SAMPLES 1000

typedef struct {
    float *w1, *b1;
    float *w2, *b2;
    float *w3, *b3;
} NeuralNetwork;

void init_network(NeuralNetwork *nn) {
    nn->w1 = (float*)malloc(INPUT_SIZE * HIDDEN1_SIZE * sizeof(float));
    nn->b1 = (float*)malloc(HIDDEN1_SIZE * sizeof(float));
    nn->w2 = (float*)malloc(HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float));
    nn->b2 = (float*)malloc(HIDDEN2_SIZE * sizeof(float));
    nn->w3 = (float*)malloc(HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->b3 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    float std_w1 = sqrt(2.0 / (INPUT_SIZE + HIDDEN1_SIZE));
    float std_w2 = sqrt(2.0 / (HIDDEN1_SIZE + HIDDEN2_SIZE));
    float std_w3 = sqrt(2.0 / (HIDDEN2_SIZE + OUTPUT_SIZE));
    
    for (int i = 0; i < INPUT_SIZE * HIDDEN1_SIZE; i++)
        nn->w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_w1;
    for (int i = 0; i < HIDDEN1_SIZE * HIDDEN2_SIZE; i++)
        nn->w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_w2;
    for (int i = 0; i < HIDDEN2_SIZE * OUTPUT_SIZE; i++)
        nn->w3[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_w3;
    
    memset(nn->b1, 0, HIDDEN1_SIZE * sizeof(float));
    memset(nn->b2, 0, HIDDEN2_SIZE * sizeof(float));
    memset(nn->b3, 0, OUTPUT_SIZE * sizeof(float));
}

void free_network(NeuralNetwork *nn) {
    free(nn->w1); free(nn->b1);
    free(nn->w2); free(nn->b2);
    free(nn->w3); free(nn->b3);
}

static inline float relu(float x) {
    return x > 0 ? x : 0;
}

void softmax(float *input, float *output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++)
        if (input[i] > max_val) max_val = input[i];
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++)
        output[i] /= sum;
}

void matmul_vec(float *matrix, float *vector, float *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++)
            sum += matrix[i * cols + j] * vector[j];
        result[i] = sum;
    }
}

void add_bias_activate(float *vector, float *bias, int size, int activation) {
    for (int i = 0; i < size; i++) {
        vector[i] += bias[i];
        if (activation == 1)
            vector[i] = relu(vector[i]);
    }
}

void forward_pass(NeuralNetwork *nn, float *input, float *output) {
    float hidden1[HIDDEN1_SIZE];
    float hidden2[HIDDEN2_SIZE];
    float logits[OUTPUT_SIZE];
    
    matmul_vec(nn->w1, input, hidden1, HIDDEN1_SIZE, INPUT_SIZE);
    add_bias_activate(hidden1, nn->b1, HIDDEN1_SIZE, 1);
    
    matmul_vec(nn->w2, hidden1, hidden2, HIDDEN2_SIZE, HIDDEN1_SIZE);
    add_bias_activate(hidden2, nn->b2, HIDDEN2_SIZE, 1);
    
    matmul_vec(nn->w3, hidden2, logits, OUTPUT_SIZE, HIDDEN2_SIZE);
    add_bias_activate(logits, nn->b3, OUTPUT_SIZE, 0);
    
    softmax(logits, output, OUTPUT_SIZE);
}

void generate_synthetic_data(float *data, int num_samples) {
    for (int i = 0; i < num_samples; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            data[i * INPUT_SIZE + j] = (float)rand() / RAND_MAX;
}

int main(void) {
    srand(time(NULL));
    
    printf("====================================\n");
    printf("Sequential Neural Network (Baseline)\n");
    printf("====================================\n");
    printf("Total Samples: %d\n", TOTAL_SAMPLES);
    printf("Network: %d -> %d -> %d -> %d\n\n",
           INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
    
    float *inputs = (float*)malloc(TOTAL_SAMPLES * INPUT_SIZE * sizeof(float));
    float *outputs = (float*)malloc(TOTAL_SAMPLES * OUTPUT_SIZE * sizeof(float));
    
    NeuralNetwork nn;
    init_network(&nn);
    generate_synthetic_data(inputs, TOTAL_SAMPLES);
    
    clock_t start = clock();
    
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        forward_pass(&nn, &inputs[i * INPUT_SIZE], &outputs[i * OUTPUT_SIZE]);
    }
    
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Performance Results:\n");
    printf("--------------------\n");
    printf("Total inference time: %.6f seconds\n", time);
    printf("Throughput: %.2f samples/second\n", TOTAL_SAMPLES / time);
    printf("Time per sample: %.6f seconds\n\n", time / TOTAL_SAMPLES);
    
    printf("Sample Prediction (first sample):\n");
    printf("----------------------------------\n");
    printf("Sample 0: [");
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        printf("%.3f", outputs[j]);
        if (j < OUTPUT_SIZE - 1) printf(", ");
    }
    printf("]\n");
    
    free(inputs);
    free(outputs);
    free_network(&nn);
    
    return 0;
}
