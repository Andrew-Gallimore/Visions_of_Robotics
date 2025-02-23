#ifndef MAKE_DEPTH_MAP_H
#define MAKE_DEPTH_MAP_H

#include <thread>
#include <atomic>
#include <cstring>

#include <vector>
#include <cmath>
#include <string>
#include "imageUtils.h"
#include "quadTree.h"
#include "Structs.h"

using namespace std;

class DepthMap {
    public:
        // Constructor/Destructor
        DepthMap(int imageWidth, int imageHeight, int gridSize, float distanceThreshold);
        ~DepthMap();

        // Public Methods
        void makeDepthMap(vector<Point>& points);
    private:
        // Variables
        int imageWidth;
        int imageHeight;
        int gridSize;
        
        float distanceThreshold;
        int gridWidth;
        int gridHeight;
        int maxPointsPerCell;
        Point* grid;
        int* cellPointCounts;

        int maxColor;
        int pixelCount;
        unsigned char *imageData;

        // Variables for debugging;
        int pointsChecked;

        // Private Methods
        void populateGrid(vector<Point>& points);
        void getLocalPoints(int pixelX, int pixelY, Point nearbyPoints[], int& numOfNearbyPoints);
};

#endif // MAKE_DEPTH_MAP_H