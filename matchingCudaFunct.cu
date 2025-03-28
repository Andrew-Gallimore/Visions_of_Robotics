#include <string>
#include <vector>
#include <cmath>
#include "utils/imageUtils.h"
#include "makeDepthMap.h"
#include "utils/timer.h"
#include "Structs.h"
#include <cuda_runtime.h>

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

void calculateDistance(float baseline, float focalLength, float disparity, float& depth) {
    if (disparity == 0) {
        depth = -1; // Handle division by zero (no depth information)
        return;
    }
    depth = (baseline * focalLength) / disparity;
}



__global__ void searchForPoints(Point* initialPoints, Point* matchPoints, int numPoints, int imageScalar, int windowSize, unsigned char* leftImage, unsigned char* rightImage, int d_imageWidth, int d_imageHeight) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPoints) {
        return;
    }

    // Getting the point for this thread
    Point point = initialPoints[index];

    // We assume the images are rectified, so we only need to search in x direction
    int itterations = 200;

    // Make left window of points to check with
    // Window should be centered around the point
    int window[81];
    int checkWindow[81];
    for(int j = 0; j < windowSize * windowSize; j++) {
        // Get the point
        int sampleX = point.x + ((j % windowSize) - (windowSize/2)) * imageScalar;
        int sampleY = point.y + ((j / windowSize) - (windowSize/2)) * imageScalar;
        if(sampleX < 0 || sampleX >= d_imageWidth || sampleY < 0 || sampleY >= d_imageHeight) {
            // Out of bounds
            window[j] = -1;
            continue;
        }
        window[j] = leftImage[sampleY * d_imageWidth + sampleX];
    }
    
    // Keep track of the points we have checked
    int pointDifferences[200];
    for(int i = 0; i < 200; i++) {
        pointDifferences[i] = -1;
    }

    // Sample all the windows for the number of itterations
    for(int i = 0; i < itterations; i++) {
        // Stop if we are out of bounds
        if ((point.x - i) < windowSize / 2 || (point.x - i) >= d_imageWidth - windowSize / 2) {
            continue;
        }

        // Get right window of point to check against
        int samplePoint = point.x - i;
        for(int j = 0; j < windowSize * windowSize; j++) {
            // Get the point
            int sampleX = samplePoint + ((j % windowSize) - (windowSize/2)) * imageScalar;
            int sampleY = point.y + ((j / windowSize) - (windowSize/2)) * imageScalar;
            if(sampleX < 0 || sampleX >= d_imageWidth || sampleY < 0 || sampleY >= d_imageHeight) {
                // Out of bounds
                checkWindow[j] = -1;
                continue;
            }
            checkWindow[j] = rightImage[sampleY * d_imageWidth + sampleX];
        }

        // Compare the two windows
        int diffSum = 0;
        for(int j = 0; j < windowSize * windowSize; j++) {
            diffSum += abs(window[j] - checkWindow[j]);
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
    matchPoints[index].x = point.x - bestIndex;
    matchPoints[index].y = point.y;
    matchPoints[index].z = bestIndex;
}





// Runs the whole matching algorithm
void matchingFunct(PPMImage* leftImage, PPMImage* rightImage, PPMImage* depthMapOutput) {
    //convertJPGToPPM("images/leftRectified2-adjusted.jpg", "images/colorTEMP.ppm");
    //convertJPGToPPM("images/rightRectified2.jpg", "images/colorTEMP2.ppm");
    //convertPPMToBW("images/colorTEMP.ppm", "images/bwTEMP.ppm");
    //convertPPMToBW("images/colorTEMP2.ppm", "images/bwTEMP2.ppm");
    //PPMImage* leftImage = readPPM("images/colorTEMP.ppm", 0);
    //PPMImage* rightImage = readPPM("images/colorTEMP2.ppm", 0);

    int numPoints = 20000;
    Point initialPoints[numPoints];
    Point matchPoints[numPoints];

    // Generate the first points
    generatePoints(initialPoints, numPoints, imageWidth, imageHeight);

    Timer matchingTimer;
    matchingTimer.start();

    // Setting up memeory in the GPU
    Point* d_initialPoints;
    Point* d_matchPoints;
    cudaMalloc(&d_initialPoints, sizeof(Point) * numPoints);
    cudaMalloc(&d_matchPoints, sizeof(Point) * numPoints);
    cudaMemcpy(d_initialPoints, initialPoints, sizeof(Point) * numPoints, cudaMemcpyHostToDevice);
    
    unsigned char* d_leftImage;
    unsigned char* d_rightImage;
    cudaMalloc(&d_leftImage, sizeof(char) * leftImage->width * leftImage->height);
    cudaMalloc(&d_rightImage, sizeof(char) * rightImage->width * rightImage->height);
    cudaMemcpy(d_leftImage, leftImage->data, sizeof(char) * leftImage->width * leftImage->height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rightImage, rightImage->data, sizeof(char) * rightImage->width * rightImage->height, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256; // Number of threads per block
    int numBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks
    
    
    // Run the kernel
    Timer kernalTimer;
    kernalTimer.start();
    searchForPoints<<<numBlocks, threadsPerBlock>>>(d_initialPoints, d_matchPoints, numPoints, 3, 9, d_leftImage, d_rightImage, leftImage->width, leftImage->height);
    kernalTimer.stop();


    // Copy the data back
    cudaMemcpy(matchPoints, d_matchPoints, sizeof(Point) * numPoints, cudaMemcpyDeviceToHost);

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
        depth = (depth + 300) / 20; // Convert to something within the range of 0-255, mostly
        initialPoints[i].z = depth;
    }


    Timer depthMapTimer;

    depthMapTimer.start();
    DepthMap depthMap(imageWidth, imageHeight, 5, 15.0);
    // DepthMap depthMap(imageWidth, imageHeight, 10, 20.0);
    depthMap.makeDepthMap(initialPoints, numPoints, depthMapOutput);
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
    printf("Kernal time: %d ms\n", (int)kernalTimer.elapsedMilliseconds());
    printf("Full Matching time: %d ms\n", (int)matchingTimer.elapsedMilliseconds());
    printf("Depth map time: %d ms\n", (int)depthMapTimer.elapsedMilliseconds());
    printf("Total time: %d ms\n", (int)(depthMapTimer.elapsedMilliseconds() + matchingTimer.elapsedMilliseconds()));
}
