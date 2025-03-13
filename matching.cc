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

bool generateGridPoints(Point array[], int& numPoints, int totalPoints, int pointIndex) {
    // point.layer = 0 => base layer, so full image size
    // point.layer = 1 => quarter of the image
    // point.layer = 2 => 1/16 of the image, etc

    // Check if we have enough space for the new points
    if(numPoints + 4 >= totalPoints) {
        printf("!!!! Not enough space for more points\n");
        return false;
    }

    // Calcuating the number of points to add
    int windowSizeWidth;
    int windowSizeHeight;
    float xStep;
    float yStep;
    int layer = -1;
    // float windowOffsetX;
    // float windowOffsetY;
    if(pointIndex == 0) {
        // Base case
        layer = 0;

        windowSizeWidth = imageWidth;
        windowSizeHeight = imageHeight;
        xStep = imageWidth / 4;
        yStep = imageHeight / 4;
        // windowOffsetX = xStep;
        // windowOffsetY = yStep;
    }else {
        // The layer the new points are at on the quadtree
        layer = array[pointIndex].layer + 1;

        // Setting a smaller window
        windowSizeWidth = imageWidth * (layer * 2);
        windowSizeHeight = imageHeight * (layer * 2);
        xStep = imageWidth / pow(2, layer + 2);
        yStep = imageHeight / pow(2, layer + 2);
        // windowOffsetX = xStep * floor(array[pointIndex].x / (xStep * 2));
        // windowOffsetY = yStep * floor(array[pointIndex].y / (yStep * 2));
    }

    
    // For all the points it needs to add
    for(int i = 0; i < 4; i++) {
        // Calculating position of the new point
        int offsetX;
        int offsetY;
        if(i % 2 == 0) {
            offsetX = -1 * xStep;
        }else {
            offsetX = xStep;
        }
        if(i / 2 == 0) {
            offsetY = -1 * yStep;
        }else {
            offsetY = yStep;
        }
        
        int x = array[pointIndex].x + offsetX;
        int y = array[pointIndex].y + offsetY;
        
        // printf("Point (%d, %d)\n", array[pointIndex].x, array[pointIndex].y);
        
        // Adding the point to the array
        array[numPoints + i + 1] = {
            x,
            y,
            -1,
            layer
        };
        
        if(x < 1 || x >= imageWidth || y < 1 || y >= imageHeight) {
            printf("Layer: %d\n", layer);
            printf("Point Index: %d\n", pointIndex);
            printf("Offset: (%d, %d)\n", offsetX, offsetY);
            printf("Parent point: (%d, %d)\n", array[pointIndex].x, array[pointIndex].y);
            printf("Point: (%d, %d)\n\n", x, y);
        }
    }

    numPoints += 4;
    return true;
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
    bool found = false;
    int itterMax = 200;
    int itterations = 0;

    // Make window of points to check with
    // Window should be centered around the point
    int window[windowSize * windowSize];
    sampleWindow(window, windowSize, imageScaler, point.x, point.y, leftImage);
    
    // Keep track of the points we have checked
    int checkedPoints[itterMax] = {-1};

    // Search for the point
    while(!found) {
        // Limit itterations
        if(itterations >= itterMax) {
            // Choose best point from the points sampled
            int bestIndex = 0;
            int bestDiff = 1000000000;
            for(int i = 0; i < itterMax; i++) {
                if(checkedPoints[i] != 0 && checkedPoints[i] < bestDiff) {
                    bestIndex = i;
                    bestDiff = checkedPoints[i];
                }
            }

            // Return the best point
            newPoint.x = point.x + dir * ((bestIndex % windowSize) - (windowSize/2));
            newPoint.y = point.y;
            newPoint.z = bestIndex;
            
            found = true;
        }

        // Get a sample of points from the image
        int sampleX = point.x + dir * itterations * imageScaler;
        int sampleY = point.y;
        // Check for out of bounds, if we hit an edge, we should return the best point
        if(sampleX <= 0 || sampleX >= imageWidth || sampleY <= 0 || sampleY >= imageHeight) {
            // Choose best point from the points sampled
            int bestIndex = 0;
            int bestDiff = 1000000000;
            for(int i = 0; i < itterMax; i++) {
                if(checkedPoints[i] != 0 && checkedPoints[i] < bestDiff) {
                    bestIndex = i;
                    bestDiff = checkedPoints[i];
                }
            }

            // Return the best point
            newPoint.x = point.x + dir * ((bestIndex % windowSize) - (windowSize/2));
            newPoint.y = point.y;
            newPoint.z = bestIndex;

            found = true;
        }

        // Sampling the window we check with
        int checkWindow[windowSize * windowSize];
        int checkDiffSum = 0;
        sampleWindow(checkWindow, windowSize, imageScaler, sampleX, sampleY, rightImage);

        // Compare the two windows
        for(int i = 0; i < windowSize * windowSize; i++) {
            checkDiffSum += abs(window[i] - checkWindow[i]);
        }

        // Keep track of the points we have checked
        checkedPoints[itterations] = checkDiffSum;

        // Debugging only...
        if(found) {
            // printf("Point/newPoint (%d, %d)\n", point.x, newPoint.x);
            // printf(" ... Found point at (%d, %d) with diff %d, itteration=%d\n", sampleX, sampleY, checkDiffSum, itterations);
        }

        itterations++;
    }
}

void calculateDistance(float spacing, float leftOffsetX, float leftCenterX, float leftFocalX, 
                                      float rightOffsetX, float rightCenterX, float rightFocalX, float& z) {
    z = -1.0 * spacing / (((rightOffsetX - rightCenterX) / rightFocalX) - ((leftOffsetX - leftCenterX) / leftFocalX));

    // printf("Calculated distance: %f\n", z);
}

// ==== Starting Function for Algorithm ====

// Runs the whole matching algorithm
int main() {
    convertJPGToPPM("images/leftRectified2.jpg", "images/colorTEMP.ppm");
    convertJPGToPPM("images/rightRectified2.jpg", "images/colorTEMP2.ppm");
    //convertPPMToBW("images/colorTEMP.ppm", "images/bwTEMP.ppm");
    //convertPPMToBW("images/colorTEMP2.ppm", "images/bwTEMP2.ppm");
    PPMImage* leftImage = readPPM("images/colorTEMP.ppm", 0);
    PPMImage* rightImage = readPPM("images/colorTEMP2.ppm", 0);

    int totalPoints = 1000;
    Point initialPoints[totalPoints];
    initialPoints[0] = {imageWidth / 2, imageHeight / 2, -1, 0};
    int numPoints = 0;
    Point matchPoints[totalPoints];

    int focusIndex = 0;
    while(numPoints + 4 < totalPoints) {
        // Generate a set of new points where its needed
        generateGridPoints(initialPoints, numPoints, totalPoints, focusIndex);

        // 
        focusIndex += 1;
    }

    printf("====\n");
    for(int i = 0; i < numPoints; i++) {
        printf("Point: (%d, %d)\n", initialPoints[i + 1].x, initialPoints[i + 1].y);
    }

    return 0;

    Timer matchingTimer;
    matchingTimer.start();

    // Search for the points in the right image
    for(int i = 0; i < numPoints; i++) {
        Point& point = initialPoints[i];
        Point newPoint = {-1000000, -1000000, -1000000};

        searchForPoint(point, newPoint, -1, 3, 9, leftImage, rightImage);

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

        // printf("Left/right point: (%d, %d)\n", leftPoint.x, rightPoint.x);

        // Calculate the distance between the two points
        float z = 0;
        // calculateDistance(spacing, leftPoint.x, calibMatrixLeft[2], calibMatrixLeft[0], 
        //                            rightPoint.x, calibMatrixRight[2], calibMatrixRight[0], z);

        // Set the z value in the point
        // matchPoints[i].z = (z - 1000) / 3;

        // printf("Distance from camera: %f\n", (z - 1100) / 1.7);
        // matchPoints[i].z = (leftPoint.x - rightPoint.x) * 50;
        matchPoints[i].z = rightPoint.z * 1.6;

        // printf("Distance from camera: %f\n", z;
    }


    Timer depthMapTimer;

    depthMapTimer.start();
    DepthMap depthMap(imageWidth, imageHeight, 6, 20.0);
    depthMap.makeDepthMap(matchPoints);
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
