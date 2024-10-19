#include <stdio.h>
#include <stdlib.h>
#include "kangaroo.h"

// Simple modular arithmetic for stepping
#define MODULUS 0xFFFFFFFFFFFFFFFFULL

// Example step function for kangaroo jumps
static uint64_t step(uint64_t value) {
    return (value * value + 1) % MODULUS;
}

void kangaroo_algorithm(uint64_t start, uint64_t end, uint64_t target) {
    // Initialize tame kangaroo starting at known point
    Kangaroo tame = { .position = start, .value = start };
    // Wild kangaroo starts at a random position between start and end
    Kangaroo wild = { .position = rand() % (end - start) + start, .value = wild.position };

    // Jumping loop
    while (tame.value != target && wild.value != target) {
        // Tame kangaroo jumps
        tame.position += step(tame.value);
        tame.value = (tame.value + tame.position) % MODULUS;

        // Wild kangaroo jumps
        wild.position += step(wild.value);
        wild.value = (wild.value + wild.position) % MODULUS;

        // Check if they land on the same position (collision)
        if (tame.position == wild.position) {
            printf("Collision detected! Potential match at position: %llu\n", tame.position);
            break;
        }
    }

    if (tame.value == target || wild.value == target) {
        printf("Target found at position: %llu\n", (tame.value == target) ? tame.position : wild.position);
    } else {
        printf("Failed to find the target within the given range.\n");
    }
}
