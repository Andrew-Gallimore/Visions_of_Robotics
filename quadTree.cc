#include "quadTree.h"

Quadtree::Quadtree(int level, int x, int y, int width, int height)
    : level(level), x(x), y(y), width(width), height(height) {
    for (int i = 0; i < 4; ++i) {
        nodes[i] = nullptr;
    }
}

Quadtree::~Quadtree() {
    clear();
}

void Quadtree::clear() {
    points.clear();
    for (int i = 0; i < 4; ++i) {
        if (nodes[i] != nullptr) {
            nodes[i]->clear();
            delete nodes[i];
            nodes[i] = nullptr;
        }
    }
}

void Quadtree::split() {
    int subWidth = width / 2;
    int subHeight = height / 2;
    int xMid = x + subWidth;
    int yMid = y + subHeight;

    nodes[0] = new Quadtree(level + 1, x, y, subWidth, subHeight);
    nodes[1] = new Quadtree(level + 1, xMid, y, subWidth, subHeight);
    nodes[2] = new Quadtree(level + 1, x, yMid, subWidth, subHeight);
    nodes[3] = new Quadtree(level + 1, xMid, yMid, subWidth, subHeight);
}

int Quadtree::getIndex(Point point) {
    int index = -1;
    int xMid = x + width / 2;
    int yMid = y + height / 2;

    bool topQuadrant = (point.y < yMid);
    bool bottomQuadrant = (point.y >= yMid);

    if (point.x < xMid) {
        if (topQuadrant) {
            index = 0;
        } else if (bottomQuadrant) {
            index = 2;
        }
    } else {
        if (topQuadrant) {
            index = 1;
        } else if (bottomQuadrant) {
            index = 3;
        }
    }

    return index;
}

void Quadtree::insert(Point point) {
    if (nodes[0] != nullptr) {
        int index = getIndex(point);
        if (index != -1) {
            nodes[index]->insert(point);
            return;
        }
    }

    points.push_back(point);

    if (points.size() > MAX_POINTS && level < MAX_LEVELS) {
        if (nodes[0] == nullptr) {
            split();
        }

        int i = 0;
        while (i < points.size()) {
            int index = getIndex(points[i]);
            if (index != -1) {
                nodes[index]->insert(points[i]);
                points.erase(points.begin() + i);
            } else {
                ++i;
            }
        }
    }
}

void Quadtree::retrieve(std::vector<Point>& returnPoints, Point point, float distanceThreshold) {
    int index = getIndex(point);
    if (index != -1 && nodes[0] != nullptr) {
        nodes[index]->retrieve(returnPoints, point, distanceThreshold);
    }

    for (const Point& p : points) {
        float distance = sqrt(pow(p.x - point.x, 2) + pow(p.y - point.y, 2));
        if (distance <= distanceThreshold) {
            returnPoints.push_back(p);
        }
    }

    // Check neighboring nodes
    retrieveFromNeighbors(returnPoints, point, distanceThreshold);
}

void Quadtree::retrieveFromNeighbors(std::vector<Point>& returnPoints, Point point, float distanceThreshold) {
    for (int i = 0; i < 4; ++i) {
        if (nodes[i] != nullptr) {
            for (const Point& p : nodes[i]->points) {
                float distance = sqrt(pow(p.x - point.x, 2) + pow(p.y - point.y, 2));
                if (distance <= distanceThreshold) {
                    returnPoints.push_back(p);
                }
            }
        }
    }
}