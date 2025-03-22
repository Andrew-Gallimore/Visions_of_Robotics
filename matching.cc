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
            layer,
            0
        };
        
        if(x < 1 || x >= imageWidth || y < 1 || y >= imageHeight) {
            printf("Point Index: %d\n\n", pointIndex);
            printf("Layer: %d\n", layer);
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
            newPoint.x = point.x + dir * bestIndex * imageScaler;
            newPoint.y = point.y;
            newPoint.z = bestIndex;
            newPoint.layer = point.layer;
            
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
            newPoint.x = point.x + dir * bestIndex * imageScaler;
            newPoint.y = point.y;
            newPoint.z = bestIndex;
            newPoint.layer = point.layer;

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
    convertJPGToPPM("images/newLeft.jpg", "images/colorTEMP.ppm");
    convertJPGToPPM("images/newRightRect.jpg", "images/colorTEMP2.ppm");
    convertPPMToBW("images/colorTEMP.ppm", "images/bwTEMP.ppm");
    convertPPMToBW("images/colorTEMP2.ppm", "images/bwTEMP2.ppm");
    // convertJPGToPPM("images/leftRectified2-adjusted.jpg", "images/bwTEMP.ppm");
    // convertJPGToPPM("images/rightRectified2.jpg", "images/bwTEMP2.ppm");

    PPMImage* leftImage = readPPM("images/bwTEMP.ppm", 0);
    PPMImage* rightImage = readPPM("images/bwTEMP2.ppm", 0);

    int requiredDiff = 50;

    int totalPoints = 4000;
    Point initialPoints[totalPoints];
    initialPoints[0] = {imageWidth / 2, imageHeight / 2, -1, 0, 0};
    int numPoints = 0;
    Point matchPoints[totalPoints];

    // Que for the points to focus on
    int que[totalPoints] = {0};
    int queStart = 0;
    int queEnd = 0;

    // For calculating depth
    // These are gotten from running calibration on the images
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

    Timer matchingTimer;
    matchingTimer.start();

    int focusIndex = 0;
    while(numPoints + 4 < totalPoints) {
        // Generate a set of new points where its needed
        // generateGridPoints(initialPoints, numPoints, totalPoints, focusIndex);

        // printf("Focus Index: %d\n", queStart);    
        // printf("Que End: %d\n", queEnd);    
        
        // Generating a new set of points where its needed in the que
        if(queStart <= queEnd) {
            generateGridPoints(initialPoints, numPoints, totalPoints, que[queStart]);
            queStart += 1;
        }
        printf("Num Points: %d\n", numPoints);

        // Search for the points in the right image
        for(int i = 0; i < 4; i++) {
            Point& point = initialPoints[numPoints - 4 + i];
            Point newPoint = {-1000000, -1000000, -1000000};

            searchForPoint(point, newPoint, -1, 3, 9, leftImage, rightImage);

            matchPoints[numPoints - 4 + i] = newPoint;
        }

        // Simulate the depth of the points
        int depth[4] = {0, 0, 0, 0};
        for(int i = 0; i < 4; i++) {
            // depth[i] = matchPoints[numPoints - 4 + i].z * 15;
            float z = 0;
            calculateDistance(60, 
                initialPoints[numPoints - 4 + i].x, 
                    calibMatrixLeft[2], calibMatrixLeft[0], 
                matchPoints[numPoints - 4 + i].x, 
                    calibMatrixRight[2], calibMatrixRight[0], z);
            
            // Set the z value in the point
            matchPoints[numPoints - 4 + i].z = z / 5;
            depth[i] = z / 5;
            // printf("Depth: %f\n", z / 5);
        }

        /* Layout:
            0 1
            2 3
        */
        // Check if we have enough difference between the neiboring points
        bool needsFocus[4] = {false, false, false, false};
        if(matchPoints[que[queStart - 1]].layer < 2) {
            // We are not at the last layer, so we should focus on the points
            needsFocus[0] = true;
            needsFocus[1] = true;
            needsFocus[2] = true;
            needsFocus[3] = true;
        }else {
            if(abs(depth[0] - depth[1]) > requiredDiff) {
                printf("Depth diff: %d\n", abs(depth[0] - depth[1]));
                // We have enough difference between the points
                // We should focus on this point
                needsFocus[0] = true;
                matchPoints[numPoints - 4 + 0].dyingBreath = 0;
            }else {
                // We don't have enough difference between the points, but we should still focus on this point an extra time, encase there is more detail
                // printf("Dying breath: %d\n", matchPoints[que[queStart - 1]].dyingBreath);
                if(matchPoints[que[queStart - 1]].dyingBreath < 1) {
                    needsFocus[0] = true;
                    matchPoints[numPoints - 4 + 0].dyingBreath += 1;
                }
            }
            if(abs(depth[2] - depth[3]) > requiredDiff) {
                // We have enough difference between the points
                // We should focus on this point
                needsFocus[1] = true;
                matchPoints[numPoints - 4 + 1].dyingBreath = 0;
            }else {
                // We don't have enough difference between the points, but we should still focus on this point an extra time, encase there is more detail
                if(matchPoints[que[queStart - 1]].dyingBreath < 1) {
                    needsFocus[1] = true;
                    matchPoints[numPoints - 4 + 1].dyingBreath += 1;
                }
            }
            if(abs(depth[0] - depth[2]) > requiredDiff) {
                // We have enough difference between the points
                // We should focus on this point
                needsFocus[2] = true;
                matchPoints[numPoints - 4 + 2].dyingBreath = 0;
            }else {
                // We don't have enough difference between the points, but we should still focus on this point an extra time, encase there is more detail
                if(matchPoints[que[queStart - 1]].dyingBreath < 1) {
                    needsFocus[2] = true;
                    matchPoints[numPoints - 4 + 2].dyingBreath += 1;
                }
            }
            if(abs(depth[1] - depth[3]) > requiredDiff) {
                // We have enough difference between the points
                // We should focus on this point
                needsFocus[3] = true;
                matchPoints[numPoints - 4 + 3].dyingBreath = 0;
            }else {
                // We don't have enough difference between the points, but we should still focus on this point an extra time, encase there is more detail
                if(matchPoints[que[queStart - 1]].dyingBreath < 1) {
                    needsFocus[3] = true;
                    matchPoints[numPoints - 4 + 3].dyingBreath += 1;
                }
            }
        }

        // Enqueue the points that need focus
        if(queEnd + 4 < totalPoints) {
            // Add the point to the que
            for(int i = 0; i < 4; i++) {
                if(needsFocus[i]) {
                    que[queEnd] = numPoints - 4 + i;
                    queEnd += 1;
                }
            }
            printf("Que End: %d\n", queEnd);
        }

        // // Calculate the depth for all the new points
        // for(int i = 0; i < 4; i++) {
        //     Point& leftPoint = initialPoints[numPoints - 4 + i];
        //     Point& rightPoint = matchPoints[numPoints -4 + i];

        //     // Calculate the distance between the two points
        //     float z = 0;
        //     calculateDistance(60.0, leftPoint.x, 256.252, 542.131, 
        //                               rightPoint.x, 252.2899, 566.5176, z);

        //     // Set the z value in the point
        //     matchPoints[numPoints - 4 + i].z = (z - 1100) / 1.7;
        // }

        // If we have enough difference between any points, we add them to the stack of points to focus on
            // Set focusIndex to the new point
        // If we don't have enough difference, we continue to the next point
        focusIndex += 1;
    }

    printf("====\n");
    // for(int i = 0; i < numPoints; i++) {
    //     printf("Point: (%d, %d)\n", initialPoints[i + 1].x, initialPoints[i + 1].y);
    // }


    // Search for the points in the right image
    // for(int i = 0; i < numPoints; i++) {
    //     Point& point = initialPoints[i];
    //     Point newPoint = {-1000000, -1000000, -1000000};

    //     searchForPoint(point, newPoint, -1, 3, 9, leftImage, rightImage);

    //     matchPoints[i] = newPoint;
    // }

    matchingTimer.stop();

    
    // Calculate the distances given the two points
    float spacing = 60.0; // mm
    
    // // Calculate the distance between the two cameras
    // for(int i = 0; i < numPoints; i++) {
    //     Point& leftPoint = initialPoints[i];
    //     Point& rightPoint = matchPoints[i];
        
    //     // printf("Left/right point: (%d, %d)\n", leftPoint.x, rightPoint.x);
        
    //     // // Calculate the distance between the two points
    //     // float z = 0;
    //     // calculateDistance(spacing, leftPoint.x, calibMatrixLeft[2], calibMatrixLeft[0], 
    //     //                            rightPoint.x, calibMatrixRight[2], calibMatrixRight[0], z);
        
    //     // // Set the z value in the point
    //     // matchPoints[i].z = z / 15;
    //     // if(i > 2000 && i < 2100) {
    //         //     printf("X diff: %d\n", leftPoint.x - rightPoint.x);
    //     //     printf("Distance from camera: %f\n\n", z / 15);
    //     // }
        
    //     // printf("Distance from camera: %f\n", (abs(z) - 1100) / 200);
    //     // matchPoints[i].z = (leftPoint.x - rightPoint.x) * 50;
    //     matchPoints[i].z = rightPoint.z * 1.6;
        
    //     // printf("Distance from camera: %f\n", z;
    // }

    for(int i = 0; i < numPoints; i++) {
        initialPoints[i].z = matchPoints[i].z;
        initialPoints[i].layer = matchPoints[i].layer;
    }
    

    Timer depthMapTimer;

    depthMapTimer.start();
    DepthMap depthMap(imageWidth, imageHeight, 50, 30.0);
    // depthMap.makeDepthMap(matchPoints, numPoints);
    depthMap.drawPoints(initialPoints, numPoints);
    depthMap.makeQuadDepthMap(initialPoints, numPoints);
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
