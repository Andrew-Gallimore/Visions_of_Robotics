#ifndef MAKE_DEPTH_MAP_H
#define MAKE_DEPTH_MAP_H

#include <thread>
#include <atomic>
#include <cstring>

#include <vector>
#include <cmath>
#include <string>
#include "imageUtils.h"

using namespace std;

struct Point {
    int x;
    int y;
    int z;
};

void makeDepthMap(vector<Point>& points, int imageWidth, int imageHeight, int gridSize, float distanceThreshold);

#endif // MAKE_DEPTH_MAP_H