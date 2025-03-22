#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>
using namespace std;

struct Point {
    int x, y, z;
    int layer = 0;
    int dyingBreath = 0;

    // Define the < operator for Point
    bool operator<(const Point& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct Round {
    int imageScalar;                    // How much to scale down the image
    int windowSize;                     // How big the window is
    int checkItterations;               // How many itterations to check for a match for each point
    int numPoints;                      // Number of points in the image
    vector<Point> initialPoints;        // Points in the first image
    vector<Point> matchPoints;          // Points *found* in the second image
};

#endif // STRUCTS_H