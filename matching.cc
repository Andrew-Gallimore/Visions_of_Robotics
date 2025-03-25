#include <string>
#include <vector>
#include <cmath>
#include "utils/imageUtils.h"
#include "makeDepthMap.h"
#include "utils/timer.h"
#include "Structs.h"

using namespace std;

// Variables
int imageWidth = 640;
int imageHeight = 480;



// ==== Helper Functions ====

// Generates the points for a round
void generatePoints(Point array[], int numPoints, int imageWidth, int imageHeight) {
    // Adding points in a grid across the image
    int xStep = imageWidth / (sqrt(numPoints) + 1);
    int yStep = imageHeight / (sqrt(numPoints) + 1);

    // Adding points randomly
    for (int i = 0; i < numPoints; i++) {
        Point newPoint = {
            rand() % imageWidth,
            rand() % imageHeight,
            0
        };
        array[i] = newPoint;
    }
}

void sampleWindow(int* window, int windowSize, int imageScale, int x, int y, PPMImage* image) {
    for(int i = 0; i < windowSize * windowSize; i++) {
        // Get the point
        int sampleX = x + ((i % windowSize) - (windowSize/2)) * imageScale;
        int sampleY = y + ((i / windowSize) - (windowSize/2)) * imageScale;
        if(sampleX < 0 || sampleX >= image->width || sampleY < 0 || sampleY >= image->height) {
            // Out of bounds
            window[i] = -1;
            continue;
        }
        window[i] = image->data[sampleY * image->width + sampleX];
    }
}

void searchForPoint(Point& point, Point& newPoint, int dir, int imageScaler, int windowSize, PPMImage* leftImage, PPMImage* rightImage) {
    // We assume the images are rectified, so we only need to search in x direction
    int itterations = 200;

    // Make left window of points to check with
    // Window should be centered around the point
    int window[windowSize * windowSize];
    int checkWindow[windowSize * windowSize];
    sampleWindow(window, windowSize, imageScaler, point.x, point.y, leftImage);
    
    // Keep track of the points we have checked
    int pointDifferences[itterations];
    fill_n(pointDifferences, itterations, -1);

    // Sample all the windows for the number of itterations
    for(int i = 0; i < itterations; i++) {
        // Stop if we are out of bounds
        if ((point.x + dir * i) < windowSize / 2 || (point.x + dir * i) >= leftImage->width - windowSize / 2) {
            continue;
        }

        // Get right window of point to check against
        int samplePoint = point.x + dir * i;
        sampleWindow(checkWindow, windowSize, imageScaler, samplePoint, point.y, rightImage);

        // Compare the two windows
        int diffSum = 0;
        for(int i = 0; i < windowSize * windowSize; i++) {
            diffSum += abs(window[i] - checkWindow[i]);
        }

        // Keep track of the points we have checked
        pointDifferences[i] = diffSum;
    }

    // Find the best point
    int bestIndex = 0;
    int bestDiff = 1000000000;
    for(int i = 0; i < itterations; i++) {
        if(pointDifferences[i] == -1) {
            continue;
        }
        if(pointDifferences[i] < bestDiff) {
            bestIndex = i;
            bestDiff = pointDifferences[i];
        }
    }

    // Return the best point
    newPoint.x = point.x + dir * bestIndex;
    newPoint.y = point.y;
    newPoint.z = bestIndex;
}

void calculateDistance(float baseline, float focalLength, float disparity, float& depth) {
    if (disparity == 0) {
        depth = -1; // Handle division by zero (no depth information)
        return;
    }
    depth = (baseline * focalLength) / disparity;
}

// ==== Starting Function for Algorithm ====

// Runs the whole matching algorithm
int main() {
    convertJPGToPPM("images/leftRectified2-adjusted.jpg", "images/colorTEMP.ppm");
    convertJPGToPPM("images/rightRectified2.jpg", "images/colorTEMP2.ppm");
    //convertPPMToBW("images/colorTEMP.ppm", "images/bwTEMP.ppm");
    //convertPPMToBW("images/colorTEMP2.ppm", "images/bwTEMP2.ppm");
    PPMImage* leftImage = readPPM("images/colorTEMP.ppm", 0);
    PPMImage* rightImage = readPPM("images/colorTEMP2.ppm", 0);

    int numPoints = 15000;
    Point initialPoints[numPoints];
    Point matchPoints[numPoints];

    // Generate the first points
    generatePoints(initialPoints, numPoints, imageWidth, imageHeight);

    Timer matchingTimer;
    matchingTimer.start();

    // Search for the points in the right image
    for(int i = 0; i < numPoints; i++) {
        Point& point = initialPoints[i];
        Point newPoint = {-1000000, -1000000, -1000000};

        searchForPoint(point, newPoint, -1, 3, 12, leftImage, rightImage);

        // printf("Point: (%d, %d), New Point: (%d, %d)\n", point.x, point.y, newPoint.x, newPoint.y);

        matchPoints[i] = newPoint;
    }

    matchingTimer.stop();

    // Got from running calibration on the images
    // fx, 0,  Ox
    // 0,  fy, Oy
    // 0,  0,  1
    float calibMatrixLeft[9] = {
        542.131, 0, 256.252,
        0, 720.9599, 274.971,
        0, 0, 1
    };
    float calibMatrixRight[9] = {
        566.5176, 0, 252.2899,
        0, 753.3832, 218.9708,
        0, 0, 1 
    };

    // Calculate the distances given the two points
    float spacing = 60.0; // mm

    // Calculate the distance between the two cameras
    for(int i = 0; i < numPoints; i++) {
        Point& leftPoint = initialPoints[i];
        Point& rightPoint = matchPoints[i];

        // Calculate disparity
        float disparity = leftPoint.x - rightPoint.x;
        if (disparity < 0) {
            disparity = 0; // Clamp to 0 to avoid negative depth
        }

        // Calculate depth
        float depth = 0;
        calculateDistance(spacing, (calibMatrixLeft[0] + calibMatrixRight[0]) / 2, disparity, depth);

        // Set the z value in the point
        depth = (depth + 300) / 20; // Convert to meters and scale to fit the image
        initialPoints[i].z = depth;

        // printf("Distance from camera: %f\n", abs(z));
        printf("Left x: %d, Right x: %d, Z: %f\n", leftPoint.x, rightPoint.x, depth);

        // printf("Distance from camera: %f\n", (z - 1100) / 1.7);
        // matchPoints[i].z = (leftPoint.x - rightPoint.x) * 50;
        // matchPoints[i].z = rightPoint.z * 1.6;

        // printf("Distance from camera: %f\n", z;
    }


    Timer depthMapTimer;

    depthMapTimer.start();
    DepthMap depthMap(imageWidth, imageHeight, 4, 15.0);
    // DepthMap depthMap(imageWidth, imageHeight, 10, 20.0);
    depthMap.makeDepthMap(initialPoints, numPoints);
    depthMapTimer.stop();
    // For 100 random-placed points, 30 grid size, 110.0 distance threshold, sigma = 4.0
    // For 400 random-placed points, 40 grid size, 40 to 60 distance threshold (at 400 points, distance between them is 36), sigma = 3 to 4
    // For 5000 random-placed points, 5 grid size, 20.0 distance threshold, sigma = 4.0

    // Higher sigma, sharper edges. Usuall 3 to 4, above is sharper, 2 is lowest
    // We want avg. number of local points to be around 7-12 (higher is more blurry)
    // We want avg. number of checked local points to not be limiting the local points it uses, so will be higher. Seams its min is dependent on how we position the points. The issues arise (when its too low) that it can't find other points to access, so gaps show up. If its too high, its just slower checking more points.

    // TODO: Make depth map not have to take a grid-sise, but instead have a better data structure that can handle the points better

    // Printing out the timeing
    printf("\n");
    printf("Matching time: %d ms\n", (int)matchingTimer.elapsedMilliseconds());
    printf("Depth map time: %d ms\n", (int)depthMapTimer.elapsedMilliseconds());
    printf("Total time: %d ms\n", (int)(depthMapTimer.elapsedMilliseconds() + matchingTimer.elapsedMilliseconds()));

    return 0;
}
