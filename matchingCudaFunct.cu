#include <string>
#include <vector>
#include <cmath>
#include "utils/imageUtils.h"
#include "utils/timer.h"
#include "Structs.h"
#include <cuda_runtime.h>

using namespace std;

__global__
void getDisparities(unsigned char* leftImage, unsigned char* rightImage, int cols, int rows, int windowSize, unsigned char* disparities, unsigned char* confidence) {
    int smallest = INT_MAX;
    int long average = 0;
    int d = 0;

    int arbitraryConfidence = 20;

    int leftX = blockDim.x * blockIdx.x + threadIdx.x;
    int leftY = blockDim.y * blockIdx.y + threadIdx.y;

    int minValue = 255;
    int maxValue = 0;
    for(int x = -1 * windowSize; x < windowSize; x++) {
        for(int y = -1 * windowSize; y < windowSize; y++) {
            int index = (leftX + x * 1) + ((leftY + y * 1) * cols);
            if(index < 0 || index >= cols * rows) {
                continue;
            }
            if (leftImage[index] < minValue) {
                minValue = leftImage[index];
            }
            if (leftImage[index] > maxValue) {
                maxValue = leftImage[index];
            }
        }
    }

    if(maxValue - minValue < arbitraryConfidence) {
        int index = (leftY * cols) + leftX;
        confidence[index] = 210;
        disparities[index] = d;
        return;
    }
    
    float maxDisparity = 200;

    for(int offset = 0; offset < leftX && offset < maxDisparity; offset += 2) {
        int comparison = 0;
        for(int x = -1 * windowSize; x <= windowSize; x++) {
            for(int y = -1 * windowSize; y <= windowSize; y++) {
                if(leftX + x >= 0 && leftX + x < cols &&
                   leftY + y >= 0 && leftY + y < rows &&
                   leftX + x - offset >= 0 && leftX + x - offset < cols)
                {
                    int leftIndex = (leftX + x) + ((leftY + y) * cols);
                    int rightIndex = ((leftX - offset) + x) + ((leftY + y) * cols);
                    
                    unsigned char leftPixel = leftImage[leftIndex];
                    unsigned char rightPixel = rightImage[rightIndex];

                    int diff = (int)leftPixel - (int)rightPixel;
                    comparison += abs(diff);
                }
            }
        }
        if(comparison < smallest) {
            smallest = comparison;
            average += smallest;
            d = (unsigned char)offset;
        }
    }

    int index = (leftY * cols) + leftX;

    int conf = 0;

    if(maxValue - minValue < arbitraryConfidence) {
        conf = 210;
    }

    int spacing = 60; // mm

    int actualDistance = (spacing * ((561.85034 + 560.63837) / 2)) / d;
    actualDistance = actualDistance / 8; // Scaling 2000 mm to about 255
    if(actualDistance > 255) actualDistance = 255;
    if(actualDistance <= 0) actualDistance = 0;
    //printf("%d ",smallest);

    if ((d > 20 && d <= 200) && smallest <= 1300)
    {
        disparities[index] = (d * 3 / maxDisparity) * 255;
        //distance[index] = (spacing * ((561.85034 + 560.63837) / 2)) / d;

        confidence[index] = conf;

    }
}

void matchingFunct(PPMImage* leftImage, PPMImage* rightImage, PPMImage* depthMapOutput) {
    // convertJPGToPPM("images/newLeft.jpg", "images/colorTEMP.ppm");
    // convertJPGToPPM("images/newRightRect.jpg", "images/colorTEMP2.ppm");
    // convertPPMToBW("images/colorTEMP.ppm", "images/bwTEMP.ppm");
    // convertPPMToBW("images/colorTEMP2.ppm", "images/bwTEMP2.ppm");
    // PPMImage* leftImage = readPPM("images/bwTEMP.ppm", 0);
    // PPMImage* rightImage = readPPM("images/bwTEMP2.ppm", 0);

    // convertJPGToPPM("opencvrect/nvcamtest_11219_s01_00000.jpg", "images/colorTEMPL_26_april.ppm");
    // convertJPGToPPM("opencvrect/nvcamtest_11114_s00_00000.jpg", "images/colorTEMPR_26_april.ppm");

    // convertJPGToPPM("opencvrect/nvcamtest_11219_s01_00000.jpg", "images/colorTEMPL_26_april.ppm");
    // convertJPGToPPM("opencvrect/nvcamtest_11219_s01_00000.jpg", "images/colorTEMPR_26_april.ppm");

    // PPMImage* leftImage = readPPM("images/colorTEMPL_26_april.ppm", 0);
    // PPMImage* rightImage = readPPM("images/colorTEMPR_26_april.ppm", 0);

    Timer totalTimer;
    Timer kernalTimer;
    
    int windowSize = 8;
    
    unsigned char disparities[leftImage->width * leftImage->height] = {0};
    int distances[leftImage->width * leftImage->height] = {0};
    unsigned char confidence[leftImage->width * leftImage->height] = {0};
    
    unsigned char* d_left; 
    unsigned char* d_right;
    unsigned char* d_disparities;
    // int* d_distances;
    unsigned char* d_confidence;
    
    cudaMalloc((void**) &d_left, leftImage->width * leftImage->height * sizeof(unsigned char)); 
    cudaMalloc((void**) &d_right, rightImage->width * rightImage->height * sizeof(unsigned char)); 
    cudaMalloc((void**) &d_disparities, rightImage->width * rightImage->height * sizeof(unsigned char)); 
    // cudaMalloc((void**) &d_distances, rightImage->width * rightImage->height * sizeof(int)); 
    cudaMalloc((void**) &d_confidence, rightImage->width * rightImage->height * sizeof(unsigned char)); 

    totalTimer.start();

    cudaMemcpy(d_left, leftImage->data, leftImage->width * leftImage->height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, rightImage->data, rightImage->width * rightImage->height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    int BLOCK_SIZE = 16;
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((leftImage->width + BLOCK_SIZE - 1) / BLOCK_SIZE, (leftImage->height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    kernalTimer.start();
    
    getDisparities<<<grid, block>>>(d_left, d_right, leftImage->width, leftImage->height, windowSize, d_disparities, d_confidence);
    cudaDeviceSynchronize();
    
    kernalTimer.stop();

    cudaMemcpy(disparities, d_disparities, leftImage->width * leftImage->height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // cudaMemcpy(distances, d_distances, leftImage->width * leftImage->height * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(confidence, d_confidence, leftImage->width * leftImage->height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // writePPM("images/depthMapP.ppm", leftImage->width, leftImage->height, 255, 0, disparities);
    // writePPM("images/confidenceP.ppm", leftImage->width, leftImage->height, 255, 0, confidence);

    // Making mixed color image for the depthMap/confidenceMap.
    // Colors are interlaced in the array: r, g, b, r, g, b, etc.
    unsigned char* mixedImage = new unsigned char[leftImage->width * leftImage->height * 3];
    for (int i = 0; i < leftImage->width * leftImage->height; i++) {
        if (confidence[i] > 0) {
            // No confidence: black
            mixedImage[i * 3 + 0] = 0;   // Red
            mixedImage[i * 3 + 1] = 0;   // Green
            mixedImage[i * 3 + 2] = 0;   // Blue
        } else {
            // Confidence: red-blue gradient based on disparity
            if (disparities[i] < 10) { // Below the range, continue blue
                mixedImage[i * 3 + 0] = 0;             // Red
                mixedImage[i * 3 + 1] = 0;             // Green
                mixedImage[i * 3 + 2] = 50;           // Blue
            } else if (disparities[i] > 90) { // Above the range, continue red
                mixedImage[i * 3 + 0] = 255;           // Red
                mixedImage[i * 3 + 1] = 0;             // Green
                mixedImage[i * 3 + 2] = 0;             // Blue
            } else { // Within the range, scale red to blue
                float normalized = (disparities[i] - 10) / 80.0f; // Normalize to range [0, 1]
                mixedImage[i * 3 + 0] = static_cast<unsigned char>(normalized * 255); // Red
                mixedImage[i * 3 + 1] = 0;                                           // Green
                mixedImage[i * 3 + 2] = static_cast<unsigned char>((1.0f - normalized) * 50); // Blue
            }
        }
    }
    // writePPM("images/mix.ppm", leftImage->width, leftImage->height, 255, 1, mixedImage);
    depthMapOutput->data = disparities;
    // distancesOutput = distances;
    delete[] mixedImage;
    
    totalTimer.stop();

    // Got from running calibration on the images
    // fx, 0,  Ox
    // 0,  fy, Oy
    // 0,  0,  1
    float calibMatrixLeft[9] = {
        //542.131, 0, 256.252,
        //0, 720.9599, 274.971,
        //0, 0, 1
        561.85034, 0.00000, 351.88312,
        0.00000, 763.06970, 200.38995,
        0.00000, 0.00000, 1.00000
    };
    float calibMatrixRight[9] = {
        //566.5176, 0, 252.2899,
        //0, 753.3832, 218.9708,
        //0, 0, 1
        560.63837, 0.00000, 377.36542,
        0.00000, 750.10541, 200.71365,
        0.00000, 0.00000, 1.00000
    };

    printf("\nKernal time: %d ms\n", (int)kernalTimer.elapsedMilliseconds());
    printf("Total time: %d ms\n", (int)totalTimer.elapsedMilliseconds());

    cudaFree(d_left); 
    cudaFree(d_right);
    cudaFree(d_disparities);
    cudaFree(d_confidence);
}