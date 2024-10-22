#ifndef EC_CUDA_H
#define EC_CUDA_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Secp256k1 prime modulus represented as an array of 64-bit words
__constant__ uint64_t P[4] = {
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL - 0x1000003D1ULL + 1ULL
};

// Generator point (G) coordinates for secp256k1
__constant__ uint64_t Gx[4] = {
    0x79BE667EF9DCBBACULL,
    0x55A06295CE870B07ULL,
    0x029BFCDB2DCE28D9ULL,
    0x59F2815B16F81798ULL
};
__constant__ uint64_t Gy[4] = {
    0x483ADA7726A3C465ULL,
    0x5DA4FBFC0E1108A8ULL,
    0xFD17B448A6855419ULL,
    0x9C47D08FFB10D4B8ULL
};

// Public key structure
struct PublicKey {
    uint64_t x[4];
    uint64_t y[4];
};

// Function to compare big integers
__device__ int cmp(uint64_t *a, uint64_t *b) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

// Function to perform modular addition for multi-word numbers
__device__ void mod_add(uint64_t *a, uint64_t *b, uint64_t *result) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t temp = (__uint128_t)a[i] + b[i] + carry;
        result[i] = (uint64_t)temp;
        carry = temp >> 64;
    }

    // Reduce result modulo p if necessary
    while (cmp(result, (uint64_t*)P) >= 0) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            __uint128_t temp = (__uint128_t)result[i] - P[i] - borrow;
            result[i] = (uint64_t)temp;
            borrow = (temp >> 64) & 1;
        }
    }
}

// Function to perform modular subtraction for multi-word numbers
__device__ void mod_sub(uint64_t *a, uint64_t *b, uint64_t *result) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t temp = (__uint128_t)a[i] - b[i] - borrow;
        result[i] = (uint64_t)temp;
        borrow = (temp >> 64) & 1;
    }

    // If the result is negative, add p to make it positive
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 4; ++i) {
            __uint128_t temp = (__uint128_t)result[i] + P[i] + carry;
            result[i] = (uint64_t)temp;
            carry = temp >> 64;
        }
    }
}

// Function to perform modular multiplication for multi-word numbers
__device__ void mod_mul(uint64_t *a, uint64_t *b, uint64_t *result) {
    uint64_t temp_result[8] = {0};

    // Perform multiplication
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            __uint128_t mul_result = (__uint128_t)a[i] * b[j] + temp_result[i + j] + carry;
            temp_result[i + j] = (uint64_t)mul_result;
            carry = mul_result >> 64;
        }
        temp_result[i + 4] = carry;
    }

    // Reduce modulo p
    // Since p is close to 2^256, we can use fast reduction
    while (cmp(temp_result + 4, (uint64_t*)P) >= 0 || temp_result[4] != 0 || temp_result[5] != 0 || temp_result[6] != 0 || temp_result[7] != 0) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            __uint128_t temp = (__uint128_t)temp_result[i] - P[i] - borrow;
            temp_result[i] = (uint64_t)temp;
            borrow = (temp >> 64) & 1;
        }
        for (int i = 4; i < 8; ++i) {
            __uint128_t temp = (__uint128_t)temp_result[i] - borrow;
            temp_result[i] = (uint64_t)temp;
            borrow = (temp >> 64) & 1;
        }
    }

    // Copy the result
    memcpy(result, temp_result, 4 * sizeof(uint64_t));

    // Final reduction
    while (cmp(result, (uint64_t*)P) >= 0) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            __uint128_t temp = (__uint128_t)result[i] - P[i] - borrow;
            result[i] = (uint64_t)temp;
            borrow = (temp >> 64) & 1;
        }
    }
}

// Function to perform modular exponentiation
__device__ void mod_exp(uint64_t *base, uint64_t *exp, uint64_t *result) {
    uint64_t one[4] = {1, 0, 0, 0};
    uint64_t zero[4] = {0, 0, 0, 0};
    uint64_t temp_base[4];
    memcpy(temp_base, base, 4 * sizeof(uint64_t));
    memcpy(result, one, 4 * sizeof(uint64_t));

    for (int i = 255; i >= 0; --i) {
        uint64_t bit = (exp[i / 64] >> (i % 64)) & 1;
        // result = result^2 mod p
        mod_mul(result, result, result);
        if (bit) {
            // result = result * base mod p
            mod_mul(result, temp_base, result);
        }
    }
}

// Function to compute modular inverse using Fermat's Little Theorem
__device__ void mod_inv(uint64_t *a, uint64_t *inv) {
    uint64_t exp[4];
    // exp = p - 2
    memcpy(exp, (uint64_t*)P, 4 * sizeof(uint64_t));
    uint64_t one[4] = {1, 0, 0, 0};
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t temp = (__uint128_t)exp[i] - one[i] - borrow;
        exp[i] = (uint64_t)temp;
        borrow = (temp >> 64) & 1;
    }
    borrow = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t temp = (__uint128_t)exp[i] - one[i] - borrow;
        exp[i] = (uint64_t)temp;
        borrow = (temp >> 64) & 1;
    }
    mod_exp(a, exp, inv);
}

// Function to perform point doubling on the elliptic curve
__device__ void point_double(uint64_t *x1, uint64_t *y1, uint64_t *x3, uint64_t *y3) {
    uint64_t s[4], s_inv[4], temp1[4], temp2[4];

    // Check if y1 is zero (point at infinity)
    bool is_zero = true;
    for (int i = 0; i < 4; ++i) {
        if (y1[i] != 0) {
            is_zero = false;
            break;
        }
    }
    if (is_zero) {
        // Point at infinity
        memset(x3, 0, 4 * sizeof(uint64_t));
        memset(y3, 0, 4 * sizeof(uint64_t));
        return;
    }

    // s = (3 * x1^2) / (2 * y1) mod p
    mod_mul(x1, x1, temp1);            // temp1 = x1^2
    uint64_t three[4] = {3, 0, 0, 0};
    mod_mul(temp1, three, temp1);      // temp1 = 3 * x1^2
    mod_add(y1, y1, temp2);            // temp2 = 2 * y1
    mod_inv(temp2, s_inv);             // s_inv = (2 * y1)^(-1) mod p
    mod_mul(temp1, s_inv, s);          // s = (3 * x1^2) * s_inv mod p

    // x3 = s^2 - 2 * x1 mod p
    mod_mul(s, s, temp1);              // temp1 = s^2
    mod_add(x1, x1, temp2);            // temp2 = 2 * x1
    mod_sub(temp1, temp2, x3);         // x3 = s^2 - 2 * x1 mod p

    // y3 = s * (x1 - x3) - y1 mod p
    mod_sub(x1, x3, temp1);            // temp1 = x1 - x3
    mod_mul(s, temp1, temp2);          // temp2 = s * (x1 - x3)
    mod_sub(temp2, y1, y3);            // y3 = temp2 - y1 mod p
}

// Function to perform point addition on the elliptic curve
__device__ void point_add(uint64_t *x1, uint64_t *y1, uint64_t *x2, uint64_t *y2, uint64_t *x3, uint64_t *y3) {
    uint64_t s[4], s_inv[4], temp1[4], temp2[4];

    // Check if x1 == x2 and y1 == y2 (point doubling)
    bool is_equal = true;
    for (int i = 0; i < 4; ++i) {
        if (x1[i] != x2[i] || y1[i] != y2[i]) {
            is_equal = false;
            break;
        }
    }
    if (is_equal) {
        point_double(x1, y1, x3, y3);
        return;
    }

    // Check if x1 == x2 (vertical line, result is point at infinity)
    is_equal = true;
    for (int i = 0; i < 4; ++i) {
        if (x1[i] != x2[i]) {
            is_equal = false;
            break;
        }
    }
    if (is_equal) {
        // Point at infinity
        memset(x3, 0, 4 * sizeof(uint64_t));
        memset(y3, 0, 4 * sizeof(uint64_t));
        return;
    }

    // s = (y2 - y1) / (x2 - x1) mod p
    mod_sub(y2, y1, temp1);            // temp1 = y2 - y1
    mod_sub(x2, x1, temp2);            // temp2 = x2 - x1
    mod_inv(temp2, s_inv);             // s_inv = (x2 - x1)^(-1) mod p
    mod_mul(temp1, s_inv, s);          // s = (y2 - y1) * s_inv mod p

    // x3 = s^2 - x1 - x2 mod p
    mod_mul(s, s, temp1);              // temp1 = s^2
    mod_sub(temp1, x1, temp2);         // temp2 = s^2 - x1
    mod_sub(temp2, x2, x3);            // x3 = temp2 - x2

    // y3 = s * (x1 - x3) - y1 mod p
    mod_sub(x1, x3, temp1);            // temp1 = x1 - x3
    mod_mul(s, temp1, temp2);          // temp2 = s * (x1 - x3)
    mod_sub(temp2, y1, y3);            // y3 = temp2 - y1 mod p
}

// CUDA kernel to compute the public key from a private key using scalar multiplication
__global__ void private_to_public(uint64_t *private_key, PublicKey *public_key) {
    uint64_t x[4] = {Gx[0], Gx[1], Gx[2], Gx[3]};
    uint64_t y[4] = {Gy[0], Gy[1], Gy[2], Gy[3]};
    uint64_t res_x[4] = {0}; // Initialize to point at infinity
    uint64_t res_y[4] = {0};
    bool initialized = false;

    for (int i = 255; i >= 0; --i) {
        if (initialized) {
            // Point doubling
            point_double(res_x, res_y, res_x, res_y);
        }

        // Check if the bit is set
        uint64_t bit = (private_key[i / 64] >> (i % 64)) & 1ULL;
        if (bit) {
            if (!initialized) {
                // Initialize result with G
                memcpy(res_x, x, 4 * sizeof(uint64_t));
                memcpy(res_y, y, 4 * sizeof(uint64_t));
                initialized = true;
            } else {
                // Point addition
                point_add(res_x, res_y, x, y, res_x, res_y);
            }
        }
    }

    memcpy(public_key->x, res_x, 4 * sizeof(uint64_t));
    memcpy(public_key->y, res_y, 4 * sizeof(uint64_t));
}

// Function to compress the public key (returns the compressed public key)
void compress_public_key(PublicKey *public_key, char *compressed_pubkey) {
    // If y is even, prefix is 02, else 03
    uint8_t prefix = (public_key->y[0] & 1) ? 0x03 : 0x02;

    // Convert x coordinate to big-endian byte array
    uint8_t x_bytes[32];
    for (int i = 0; i < 4; ++i) {
        uint64_t word = public_key->x[3 - i];
        x_bytes[i * 8 + 0] = (word >> 56) & 0xFF;
        x_bytes[i * 8 + 1] = (word >> 48) & 0xFF;
        x_bytes[i * 8 + 2] = (word >> 40) & 0xFF;
        x_bytes[i * 8 + 3] = (word >> 32) & 0xFF;
        x_bytes[i * 8 + 4] = (word >> 24) & 0xFF;
        x_bytes[i * 8 + 5] = (word >> 16) & 0xFF;
        x_bytes[i * 8 + 6] = (word >> 8) & 0xFF;
        x_bytes[i * 8 + 7] = word & 0xFF;
    }

    // Build the compressed public key
    sprintf(compressed_pubkey, "%02X", prefix);
    for (int i = 0; i < 32; ++i) {
        sprintf(compressed_pubkey + 2 + i * 2, "%02X", x_bytes[i]);
    }
}

void convert_private_to_public(uint64_t *private_key, PublicKey *public_key) {
    uint64_t *d_private_key;
    PublicKey *d_public_key;

    cudaMalloc(&d_private_key, sizeof(uint64_t) * 4);
    cudaMalloc(&d_public_key, sizeof(PublicKey));

    cudaMemcpy(d_private_key, private_key, sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);

    private_to_public<<<1, 1>>>(d_private_key, d_public_key);

    cudaMemcpy(public_key, d_public_key, sizeof(PublicKey), cudaMemcpyDeviceToHost);

    cudaFree(d_private_key);
    cudaFree(d_public_key);
}

#endif // EC_CUDA_H
