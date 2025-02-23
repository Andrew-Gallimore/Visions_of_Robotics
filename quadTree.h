#ifndef QUADTREE_H
#define QUADTREE_H

#include <vector>
#include <cmath>
#include "Structs.h"

class Quadtree {
public:
    Quadtree(int level, int x, int y, int width, int height);
    ~Quadtree();
    void clear();
    void insert(Point point);
    void retrieve(std::vector<Point>& returnPoints, Point point, float distanceThreshold);

private:
    int MAX_POINTS = 4;
    int MAX_LEVELS = 5;

    int level;
    int x, y, width, height;
    std::vector<Point> points;
    Quadtree* nodes[4];

    void split();
    int getIndex(Point point);
    void retrieveFromNeighbors(std::vector<Point>& returnPoints, Point point, float distanceThreshold);
};

#endif // QUADTREE_H