// deploy_kangaroos.cpp
#include "deploy_kangaroos.h"
#include <iostream>
#include "secp256k1/SECP256k1.h"
#include "secp256k1/Point.h"
#include "secp256k1/Int.h"
#include <random>

#ifndef KANGAROO_BATCH_SIZE
#define KANGAROO_BATCH_SIZE 256
#endif

// Function to deploy kangaroos for processing the private keys in the batch
void deploy_kangaroos(const std::vector<Int>& kangaroo_batch) {
    Secp256K1 secp; // Initialize the SECP256K1 context using the default constructor

    Point target_key; // Assume target_key is defined somewhere globally or passed in

    // Setup a random number generator for the jumps
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(1, 10000); // Random range for jumps

    // Process each private key in the batch
    for (const auto& base_key : kangaroo_batch) {
        Int current_key = base_key;

        // Perform a fixed number of jumps for this example
        const int KANGAROO_JUMPS = 256;
        for (int jump = 0; jump < KANGAROO_JUMPS; ++jump) {
            // Compute the corresponding public key
            Point current_pubkey = secp.ComputePublicKey(&current_key);

            // Check if the current public key matches the target key
            if (current_pubkey.equals(target_key)) {
                std::cout << "[+] Target Key Found: " << current_key.GetBase16() << std::endl;
                return; // Stop if target is found
            }

            // Make a random jump by adding a random value to the private key
            Int jump_value;
            jump_value.SetInt64(dis(gen)); // Random value for the jump
            current_key.Add(&jump_value);
        }
    }
}
