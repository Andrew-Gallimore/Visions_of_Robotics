#include <string>
#include <vector>
#include <cmath>
#include "imageUtils.h"
#include "makeDepthMap.h"
#include "timer.h"
#include "Structs.h"

using namespace std;


// Define structures for a round and a point
// struct Point {
//     int x;
//     int y;
//     int z;
// };
// struct Round {
//     int imageScalar;                    // How much to scale down the image
//     int windowSize;                     // How big the window is
//     int checkItterations;               // How many itterations to check for a match for each point
//     int numPoints;                      // Number of points in the image
//     vector<Point> initialPoints;        // Points in the first image
//     vector<Point> matchPoints;          // Points *found* in the second image
// };

// Variables
vector<Round> rounds;
int imageWidth = 720;
int imageHeight = 480;




// ==== Helper Functions ====

// Generates the points for a round
vector<Point> generatePoints(int numPoints, int imageWidth, int imageHeight) {
    vector<Point> newPoints;

    // Adding points in a grid across the image
    int xStep = imageWidth / (sqrt(numPoints) + 1);
    int yStep = imageHeight / (sqrt(numPoints) + 1);
    
    // for(int i = 0; i < sqrt(numPoints); i++) {
    //     for(int j = 0; j < sqrt(numPoints); j++) {
    //         Point newPoint = {
    //             (i * xStep) + xStep,
    //             (j * yStep) + yStep,
    //             0
    //         };
    //         newPoints.push_back(newPoint);
    //     }
    // }

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

// A single round of searching in the algorithm
void runRound(Round& round) {
    // Loop over all the points in the round
    for(int i = 0; i < round.numPoints; i++) {
        Point& point = round.initialPoints[i];
        Point newPoint = {-1000000, -1000000, -1000000};
        
        // Determine the direction to check in (-1 = left, 1 = right)
        // If on left side of screen, search left
        int dir = 1;
        if (point.x < round.windowSize / 2) {
            dir = -1;
        }

        // Search for the point in the image (limited by number of itterations)
        for (int j = 0; j < round.checkItterations; j++) {
            // TODO: Implement the search for the point

            // TEMPORARY
            newPoint.x = point.x - 2;
            newPoint.y = point.y - 2;
            newPoint.z = point.z;
        }

        // Put the found point into the matchPoints
        round.matchPoints.push_back(newPoint);
    }
}

// Generates the rounds we want for the algorithm
void initializeRounds(vector<Round>& rounds) {
    // Defines the rounds for algorithm
    rounds.push_back({
        8,  // 1/8th size
        4,  // 4x4 window
        30, // 30 Itterations on check
        5000,  // 4 points
        generatePoints(5000, imageWidth, imageHeight),
        {}
    });
    rounds.push_back({
        1,  // Full size
        16, // 16x16 window
        16, // 16 Itterations on check
        5000,  // 4 points
        {},
        {}
    });
}

// ==== Starting Function for Algorithm ====

// Runs the whole matching algorithm
int main() {
    // TODO: Read in the images
    initializeRounds(rounds);

    // Print out the initial points
    // printf("Initial points: \n");
    // for (int i = 0; i < rounds[0].numPoints; i++) {
    //     printf("P%d: (%d, %d, %d) \n", i, rounds[0].initialPoints[i].x, rounds[0].initialPoints[i].y, rounds[0].initialPoints[i].z);
    // }
    // printf("\n");

    // Loop over all the rounds
    for (int i = 0; i < (int)rounds.size(); i++) {
        // Run the round
        runRound(rounds[i]);

        // TODO: Calculate distances between points
        // TODO: Maybe do some filtering of points/add some more, etc.
            // - If so, should depend on a parameter in the round if we do

        // Put results into the next round
        if (i < (int)rounds.size() - 1) {
            rounds[i + 1].initialPoints = rounds[i].matchPoints;
        }
    }

    // Manually setting z values for making the gradient depth map
    // rounds[1].matchPoints[0].z = 0;
    // rounds[1].matchPoints[1].z = 50;
    // rounds[1].matchPoints[2].z = 200;
    // rounds[1].matchPoints[3].z = 255;
    
    // Filling the z values with random values
    for (int i = 0; i < rounds[1].numPoints; i++) {
        rounds[1].matchPoints[i].z = rand() % 256;
    }

    // TODO: Make a gradient depth map from the points
    Timer timer;
    timer.start();
    DepthMap depthMap(imageWidth, imageHeight, 5, 20.0);
    depthMap.makeDepthMap(rounds[1].matchPoints);
    timer.stop();
    printf("Time to make depth map: %f ms\n", timer.elapsedMilliseconds());
    // For 100 random-placed points, 30 grid size, 110.0 distance threshold, sigma = 4.0
    // For 400 random-placed points, 40 grid size, 40 to 60 distance threshold (at 400 points, distance between them is 36), sigma = 3 to 4
    // For 5000 random-placed points, 5 grid size, 20.0 distance threshold, sigma = 4.0

    // Higher sigma, sharper edges. Usuall 3 to 4, above is sharper, 2 is lowest
    // We want avg. number of local points to be around 7-12 (higher is more blurry)
    // We want avg. number of checked local points to not be limiting the local points it uses, so will be higher. Seams its min is dependent on how we position the points. The issues arise (when its too low) that it can't find other points to access, so gaps show up. If its too high, its just slower checking more points.



    // Print out the results
    // for (int i = 0; i < (int)rounds.size(); i++) {
    //     printf("Round %d found points: \n", i + 1);
    //     for (int j = 0; j < rounds[i].numPoints; j++) {
    //         printf("P%d: (%d, %d, %d) \n", j, rounds[i].matchPoints[j].x, rounds[i].matchPoints[j].y, rounds[i].matchPoints[j].z);
    //     }
    //     printf("\n");
    // }

    return 0;
}