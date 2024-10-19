// kangaroo.h
#ifndef KANGAROO_H
#define KANGAROO_H

#include <vector>
#include <thread>
#include <mutex>
#include "secp256k1/Point.h"

// Forward declaration to integrate elliptic curve functions from keyhunt.cpp
namespace Keyhunt {
    extern Point AddDirect(const Point& p1, const Point& p2);
}

class Kangaroo {
public:
    Kangaroo();
    ~Kangaroo();

    void setupKangaroo(int thread_id, Point start_point);
    void run();
    static void kangarooThread(Kangaroo* instance, int thread_id);

private:
    int thread_id;
    Point position;
    std::vector<Point> random_steps;
    std::mutex position_mutex;

    void fetchGiantSteps(int batch_size);
    void makeJump();
};

#endif // KANGAROO_H
