#include <string>
#include <vector>
#include <cmath>
#include "utils/imageUtils.h"
#include "makeDepthMap.h"
#include "utils/timer.h"
#include "Structs.h"

using namespace std;

// Variables
vector<Round> rounds;
int imageWidth = 640;
int imageHeight = 480;



// ==== Helper Functions ====

// Generates the points for a round
vector<Point> generatePoints(int numPoints, int imageWidth, int imageHeight) {
    vector<Point> newPoints;

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
        newPoints.push_back(newPoint);
    }

    // Returning the points
    return newPoints;
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
    // printf("Searching for point at (%d, %d)\n", point.x, point.y);
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

        // If the difference is small enough, we have found the point
        // if(checkDiffSum < 10000) {
        //     newPoint.x = sampleX;
        //     newPoint.y = sampleY;
        //     newPoint.z = point.z;
            
        //     printf(" ... Found point at (%d, %d) with diff %d, itteration=%d\n", sampleX, sampleY, checkDiffSum, itterations);
        //     found = true;
        // }

        if(found) {
            // printf("Point/newPoint (%d, %d)\n", point.x, newPoint.x);
            // printf(" ... Found point at (%d, %d) with diff %d, itteration=%d\n", sampleX, sampleY, checkDiffSum, itterations);
        }

        itterations++;
    }
}

// A single round of searching in the algorithm
void runRound(Round& round, PPMImage* leftImage, PPMImage* rightImage) {
    // Loop over all the points in the round
    for(int i = 0; i < round.numPoints; i++) {
        Point& point = round.initialPoints[i];
        Point newPoint = {-1000000, -1000000, -1000000};
        
        // Determine the direction to check in (-1 = left, 1 = right)
        // If on left side of screen, search left
        int dir = -1;
        // if (point.x < round.windowSize / 2) {
        //     dir = -1;
        // }

        searchForPoint(point, newPoint, dir, round.imageScalar, round.windowSize, leftImage, rightImage);

        // printf("Point %d: (%d, %d) -> (%d, %d)\n", i, point.x, point.y, newPoint.x, newPoint.y);

        // Put the found point into the matchPoints
        round.matchPoints.push_back(newPoint);
    }

    // printf("Distance between points 1: %d\n", round.initialPoints[0].x - round.matchPoints[0].x);
}

// Generates the rounds we want for the algorithm
void initializeRounds(vector<Round>& rounds) {
    // Defines the rounds for algorithm
    rounds.push_back({
        3,  // 1/8th size
        9,  // 4x4 window
        30, // 30 Itterations on check
        12000,  // 4 points
        generatePoints(12000, imageWidth, imageHeight),
        {}
    });
    // rounds.push_back({
    //     1,  // Full size
    //     9, // 16x16 window
    //     16, // 16 Itterations on check
    //     12000,  // 4 points
    //     {},
    //     {}
    // });
}

void calculateDistance(float spacing, float leftOffsetX, float leftCenterX, float leftFocalX, 
                                      float rightOffsetX, float rightCenterX, float rightFocalX, float& z) {
    z = -1.0 * spacing / (((rightOffsetX - rightCenterX) / rightFocalX) - ((leftOffsetX - leftCenterX) / leftFocalX));

    // printf("Calculated distance: %f\n", z);
}

// ==== Starting Function for Algorithm ====

// Runs the whole matching algorithm
int main() {
    // convertJPGToPPM("left.jpg", "left.ppm");
    PPMImage* leftImage = readPPM("leftBW.ppm", 0);
    PPMImage* rightImage = readPPM("rightBW.ppm", 0);
    initializeRounds(rounds);

    // Print out the initial points
    // printf("Initial points: \n");
    // for (int i = 0; i < rounds[0].numPoints; i++) {
    //     printf("P%d: (%d, %d, %d) \n", i, rounds[0].initialPoints[i].x, rounds[0].initialPoints[i].y, rounds[0].initialPoints[i].z);
    // }
    // printf("\n");

    Timer roundsTimer;

    roundsTimer.start();

    // Loop over all the rounds
    for (int i = 0; i < (int)rounds.size(); i++) {
        // Run the round
        runRound(rounds[i], leftImage, rightImage);
        printf(" >>> Finished round %d\n", i);

        // TODO: Calculate distances between points
        // TODO: Maybe do some filtering of points/add some more, etc.
            // - If so, should depend on a parameter in the round if we do
        
        // for(int j = 0; j < rounds[i].numPoints; j++) {
        //     printf("Distance between points 2 %d: %d\n", j, rounds[i].initialPoints[j].x - rounds[i].matchPoints[j].x);
        // }
        // printf("Distance between points 2: %f\n", rounds[0].initialPoints[0].x - rounds[0].matchPoints[0].x);

        // Put results into the next round
        if (i < (int)rounds.size() - 1) {
            rounds[i + 1].initialPoints = rounds[i].matchPoints;
        }
    }

    roundsTimer.stop();

    int lastRound = rounds.size() - 1;

    // Got from running calibration on the images
    // fx, 0,  Ox
    // 0,  fy, Oy
    // 0,  0,  1
    float calibMatrixLeft[9] = {
        561.85034, 0.00000, 351.88312, 
        0.00000, 763.06970, 200.38995, 
        0.00000, 0.00000, 1.00000
    };
    float calibMatrixRight[9] = {
        560.63837, 0.00000, 377.36542, 
        0.00000, 750.10541, 200.71365, 
        0.00000, 0.00000, 1.00000 
    };

    // Calculate the distances given the two points
    float spacing = 60.0; // mm

    // Calculate the distance between the two cameras
    for(int i = 0; i < rounds[lastRound].numPoints; i++) {
        Point& leftPoint = rounds[lastRound].initialPoints[i];
        Point& rightPoint = rounds[lastRound].matchPoints[i];

        // printf("Left/right point: (%d, %d)\n", leftPoint.x, rightPoint.x);

        // Calculate the distance between the two points
        float z = 0;
        // calculateDistance(spacing, leftPoint.x, calibMatrixLeft[2], calibMatrixLeft[0], 
        //                            rightPoint.x, calibMatrixRight[2], calibMatrixRight[0], z);

        // Set the z value in the point
        // rounds[lastRound].matchPoints[i].z = (z - 1000) / 3;

        // printf("Distance from camera: %f\n", (z - 1100) / 1.7);
        // rounds[lastRound].matchPoints[i].z = (leftPoint.x - rightPoint.x) * 50;
        rounds[lastRound].matchPoints[i].z = rightPoint.z * 2;

        // printf("Distance from camera: %f\n", z;
    }

    
    // Filling the z values with random values
    // for (int i = 0; i < rounds[lastRound].numPoints; i++) {
    //     rounds[lastRound].matchPoints[i].z = rand() % 256;
    // }


    Timer depthMapTimer;

    depthMapTimer.start();
    DepthMap depthMap(imageWidth, imageHeight, 4, 15.0);
    depthMap.makeDepthMap(rounds[lastRound].matchPoints);
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
    printf("Rounds time: %d ms\n", (int)roundsTimer.elapsedMilliseconds());
    printf("Depth map time: %d ms\n", (int)depthMapTimer.elapsedMilliseconds());
    printf("Total time: %d ms\n", (int)(depthMapTimer.elapsedMilliseconds() + roundsTimer.elapsedMilliseconds()));

    return 0;
}