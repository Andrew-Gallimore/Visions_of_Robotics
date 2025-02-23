#include <string>
#include <vector>
#include <cmath>

using namespace std;


// Define structures for a round and a point
struct Point {
    int x;
    int y;
    int z;
};
struct Round {
    int imageScalar;                    // How much to scale down the image
    int windowSize;                     // How big the window is
    int checkItterations;               // How many itterations to check for a match for each point
    int numPoints;                      // Number of points in the image
    vector<Point> initialPoints;        // Points in the first image
    vector<Point> matchPoints;          // Points *found* in the second image
};

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
    
    for(int i = 0; i < sqrt(numPoints); i++) {
        for(int j = 0; j < sqrt(numPoints); j++) {
            Point newPoint = {
                (i * xStep) + xStep,
                (j * yStep) + yStep,
                0
            };
            newPoints.push_back(newPoint);
        }
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
        4,  // 4 points
        generatePoints(4, imageWidth, imageHeight),
        {}
    });
    rounds.push_back({
        1,  // Full size
        16, // 16x16 window
        16, // 16 Itterations on check
        4,  // 4 points
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
    printf("Initial points: \n");
    for (int i = 0; i < rounds[0].numPoints; i++) {
        printf("P%d: (%d, %d, %d) \n", i, rounds[0].initialPoints[i].x, rounds[0].initialPoints[i].y, rounds[0].initialPoints[i].z);
    }
    printf("\n");

    // Loop over all the rounds
    for (int i = 0; i < (int)rounds.size(); i++) {
        // Run the round
        runRound(rounds[i]);

        // Put results into the next round
        if (i < (int)rounds.size() - 1) {
            rounds[i + 1].initialPoints = rounds[i].matchPoints;
        }
    }

    // Print out the results
    for (int i = 0; i < (int)rounds.size(); i++) {
        printf("Round %d found points: \n", i + 1);
        for (int j = 0; j < rounds[i].numPoints; j++) {
            printf("P%d: (%d, %d, %d) \n", j, rounds[i].matchPoints[j].x, rounds[i].matchPoints[j].y, rounds[i].matchPoints[j].z);
        }
        printf("\n");
    }

    return 0;
}