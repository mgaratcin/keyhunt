// kangaroo.cpp
#include "kangaroo.h"
#include "keyhunt.h" // Includes Keyhunt's cryptographic functions

#include <iostream>
#include <chrono>

Kangaroo::Kangaroo() : thread_id(0) {
}

Kangaroo::~Kangaroo() {
}

void Kangaroo::setupKangaroo(int id, Point start_point) {
    thread_id = id;
    position = start_point;
}

void Kangaroo::fetchGiantSteps(int batch_size) {
    // Fetch a batch of random giant steps using elliptic curve points
    random_steps.clear();
    for (int i = 0; i < batch_size; ++i) {
        // Generate random point (as an example, use AddDirect with different points)
        Point random_step = Keyhunt::AddDirect(position, position); // Double the position as a simple step
        random_steps.push_back(random_step);
    }
}

void Kangaroo::makeJump() {
    // Perform a jump using one of the random steps
    std::lock_guard<std::mutex> lock(position_mutex);
    if (!random_steps.empty()) {
        position = Keyhunt::AddDirect(position, random_steps.back());
        random_steps.pop_back();
    }
}

void Kangaroo::kangarooThread(Kangaroo* instance, int thread_id) {
    // Set up the kangaroo with an initial elliptic curve point (start position)
    Point start_point; // Define a proper starting point based on the application
    instance->setupKangaroo(thread_id, start_point);
    instance->fetchGiantSteps(1024); // Fetch 1,024 steps

    for (int i = 0; i < 1024; ++i) {
        instance->makeJump();
        // Optional: Add some delay or condition to check for solution
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Thread " << thread_id << " finished at position: " << instance->position.ToString() << std::endl;
}

void Kangaroo::run() {
    // Create two threads for the kangaroo algorithm
    std::thread kangaroo1(Kangaroo::kangarooThread, this, 1);
    std::thread kangaroo2(Kangaroo::kangarooThread, this, 2);

    // Join threads to the main thread
    kangaroo1.join();
    kangaroo2.join();
}
