#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include "utils/imageUtils.h"
#include "utils/timer.h"
#include "Structs.h"

using namespace std;

__global__
void getDisparities(unsigned char* leftImage, unsigned char* rightImage, int cols, int rows, int windowSize, unsigned char* disparities) {
    int smallest = INT_MAX;
    int d = 0;

    int leftX = blockDim.x * blockIdx.x + threadIdx.x;
    int leftY = blockDim.y * blockIdx.y + threadIdx.y;

    for(int offset = 0; offset < leftX; offset++) {
	
	int comparison = 0;

	for(int x = -1 * windowSize; x < windowSize; x++) {
	    for(int y = -1 * windowSize; y < windowSize; y++) {
		int leftIndex = (leftX + x) + ((leftY + y) * cols);
		int rightIndex = ((leftX - offset) + x) + ((leftY + y) * cols);
		
		if(leftIndex < 0 || leftIndex >= cols * rows || rightIndex < 0 || rightIndex >=  cols * rows) {
			continue;
		}
		
		int diff = leftImage[leftIndex] - rightImage[rightIndex];
		comparison += diff * diff;
	    }
	}
	
	if(comparison < smallest) {
	    smallest = comparison;
	    d = offset;
	}
    }

    int index = (leftY * cols) + leftX;

    disparities[index] = d;
}

// ==== Starting Function for Algorithm ====

// Runs the whole matching algorithm
int main() {
    //convertJPGToPPM("images/leftRectified2.jpg", "images/colorTEMP.ppm");
    //convertJPGToPPM("images/rightRectified2.jpg", "images/colorTEMP2.ppm");
    //convertPPMToBW("images/colorTEMP.ppm", "images/bwTEMP.ppm");
    //convertPPMToBW("images/colorTEMP2.ppm", "images/bwTEMP2.ppm");
    PPMImage* leftImage = readPPM("images/leftBW.ppm", 0);
    PPMImage* rightImage = readPPM("images/rightBW.ppm", 0);

    Timer matchingTimer;

    int windowSize;
    cout << "(4=9x9,5=11x11,etc..)\nEnter windowSize: ";
    cin >> windowSize;

    unsigned char disparities[leftImage->width * leftImage->height] = {0};

    unsigned char* d_left; 
    unsigned char* d_right;
    unsigned char* d_disparities;

    cudaMalloc((void**) &d_left, leftImage->width * leftImage->height * sizeof(unsigned char)); 
    cudaMalloc((void**) &d_right, rightImage->width * rightImage->height * sizeof(unsigned char)); 
    cudaMalloc((void**) &d_disparities, rightImage->width * rightImage->height * sizeof(unsigned char)); 

    cudaMemcpy(d_left, leftImage->data, leftImage->width * leftImage->height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, rightImage->data, rightImage->width * rightImage->height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 16;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((leftImage->width + BLOCK_SIZE - 1) / BLOCK_SIZE, (leftImage->height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matchingTimer.start();

    getDisparities<<<grid, block>>>(d_left, d_right, leftImage->width, leftImage->height, windowSize, d_disparities);
    cudaDeviceSynchronize();
    
    matchingTimer.stop();

    cudaMemcpy(disparities, d_disparities, leftImage->width * leftImage->height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
   
    
    writePPM("depthMapP.ppm", leftImage->width, leftImage->height, 255, 0, disparities);
    
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

    printf("\nMatching time: %d ms\n", (int)matchingTimer.elapsedMilliseconds());

    return 0;
}
