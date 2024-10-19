#ifndef KANGAROO_H
#define KANGAROO_H

#include <stdint.h>

// Struct to represent a kangaroo position in the search
typedef struct {
    uint64_t position;
    uint64_t value;
} Kangaroo;

// Function to run the Pollard's kangaroo algorithm
void kangaroo_algorithm(uint64_t start, uint64_t end, uint64_t target);

#endif // KANGAROO_H
