#include "timer.h"

Timer::Timer() : running(false) {}

void Timer::start() {
    startTime = std::chrono::high_resolution_clock::now();
    running = true;
}

void Timer::stop() {
    endTime = std::chrono::high_resolution_clock::now();
    running = false;
}

double Timer::elapsedMilliseconds() const {
    std::chrono::time_point<std::chrono::high_resolution_clock> endTimePoint;

    if (running) {
        endTimePoint = std::chrono::high_resolution_clock::now();
    } else {
        endTimePoint = endTime;
    }

    return std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTime).count();
}