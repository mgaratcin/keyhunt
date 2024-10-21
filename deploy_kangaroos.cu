#include "deploy_kangaroos.h"
#include <iostream>
#include "secp256k1/SECP256k1.h"
#include "secp256k1/Point.h"
#include "secp256k1/Int.h"
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define targetKey = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
#define NUM_JUMPS 512

// CUDA kernel to process kangaroo batches
__global__ void deployKangarooKernel(const Int* kangarooBatch, int batchSize, Point targetKey, Int* results, bool* matchFound) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        // Initialize SECP256K1 context
        Secp256K1 secp;

        // Get the base key for this thread's kangaroo
        Int baseKey = kangarooBatch[idx];

        // Initialize random number generator for this thread
        curandState state;
        curand_init(1234, idx, 0, &state); // Seed can be adjusted

        // Compute initial public key from base key
        Point currentPoint = secp.multiply(baseKey, secp.getGenerator());

        // Perform NUM_JUMPS forward jumps
        for (int i = 0; i < NUM_JUMPS; ++i) {
            // Generate a random jump distance
            uint64_t jumpDistance = curand(&state) % 10000 + 1;

            // Move the current point forward by the jump distance
            currentPoint = secp.add(currentPoint, secp.multiply(jumpDistance, secp.getGenerator()));

            // Check if currentPoint matches the targetKey
            if (currentPoint == targetKey) {
                matchFound[idx] = true; // Mark as found
                results[idx] = baseKey; // Store the matching base key
                return; // Exit early if a match is found
            }
        }

        // If no match is found, set result to an invalid value
        matchFound[idx] = false;
        results[idx] = Int(-1); // Indicates no match
    }
}

// Function to deploy kangaroos on the GPU
void deploy_kangaroos_gpu(const std::vector<Int>& kangarooBatch) {
    int batchSize = kangarooBatch.size();

    // Allocate memory on the GPU
    Int* d_kangarooBatch;
    Int* d_results;
    bool* d_matchFound;
    cudaMalloc(&d_kangarooBatch, batchSize * sizeof(Int));
    cudaMalloc(&d_results, batchSize * sizeof(Int));
    cudaMalloc(&d_matchFound, batchSize * sizeof(bool));

    // Copy data from host to device
    cudaMemcpy(d_kangarooBatch, kangarooBatch.data(), batchSize * sizeof(Int), cudaMemcpyHostToDevice);

    // Define the target key (assumed to be defined globally or passed to the function)
    Point targetKey; // Initialize this with the correct value

    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (batchSize + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    deployKangarooKernel<<<gridSize, blockSize>>>(d_kangarooBatch, batchSize, targetKey, d_results, d_matchFound);

    // Copy results back to host
    std::vector<Int> results(batchSize);
    std::vector<bool> matchFound(batchSize);
    cudaMemcpy(results.data(), d_results, batchSize * sizeof(Int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matchFound.data(), d_matchFound, batchSize * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_kangarooBatch);
    cudaFree(d_results);
    cudaFree(d_matchFound);

    // Process results on the host
    for (int i = 0; i < batchSize; ++i) {
        if (matchFound[i]) {
            std::cout << "Match found for base key: " << results[i] << std::endl;
        } else {
            std::cout << "No match found for base key: " << kangarooBatch[i] << std::endl;
        }
    }
}
