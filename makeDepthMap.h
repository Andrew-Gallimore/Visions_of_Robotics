#ifndef MAKE_DEPTH_MAP_H
#define MAKE_DEPTH_MAP_H

#include <thread>
#include <atomic>
#include <cstring>

#include <vector>
#include <cmath>
#include <string>
#include "utils/imageUtils.h"
#include "Structs.h"

using namespace std;

class DepthMap {
    public:
        // Constructor/Destructor
        DepthMap(int imageWidth, int imageHeight, int gridSize, float distanceThreshold);
        ~DepthMap();

        // Public Methods
        void makeDepthMap(Point points[], int numPoints);
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
        void populateGrid(Point points[], int numPoints);
        void getLocalPoints(int pixelX, int pixelY, Point nearbyPoints[], int& numOfNearbyPoints);
};

#endif // MAKE_DEPTH_MAP_H