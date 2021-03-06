#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <memory.h>

#define max(x, y)  x > y ? x : y
#define min(x, y)  x < y ? x : y
#define random(x)  (rand() % x)

using namespace std::chrono;
using std::exp;
using std::sqrt;

class Clk
{
public:
    Clk(std::string name)
    {
        updateStartNow();
        this->name = name;
    }

    ~Clk() = default;

    void updateStartNow()
    {
        _start = high_resolution_clock::now();
    }

    long long getDuration()
    {
        return duration_cast<microseconds>(high_resolution_clock::now() - _start).count();
    }

    static void printDuration(Clk &c)
    {
        long long dura = c.getDuration();
        std::cout << c.name << ":" << dura << "ms" << std::endl;
    }

private:
    time_point<high_resolution_clock> _start;
    std::string name;
};

void foo();

#endif