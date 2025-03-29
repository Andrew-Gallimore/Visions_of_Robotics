#include "makeDepthMapCuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

DepthMap::DepthMap(int imageWidth, int imageHeight, int gridSize, float distanceThreshold) {
    this->imageWidth = imageWidth;
    this->imageHeight = imageHeight;
    this->gridSize = gridSize;
    this->distanceThreshold = distanceThreshold;
    
    this->maxColor = 255;
    this->pixelCount = imageWidth * imageHeight;
    this->imageData = (unsigned char *)malloc(pixelCount * sizeof(unsigned char));
    
    // Create a grid for spatial partitioning
    this->gridWidth = (imageWidth + gridSize - 1) / gridSize;
    this->gridHeight = (imageHeight + gridSize - 1) / gridSize;
    this->maxPointsPerCell = 500; // Assuming a maximum of x points per cell

    this->grid = new Point[this->gridWidth * this->gridHeight * maxPointsPerCell];
    this->cellPointCounts = new int[this->gridWidth * this->gridHeight]();
}

DepthMap::~DepthMap() {
    free(imageData);
    delete[] grid;
    delete[] cellPointCounts;
}

void DepthMap::populateGrid(Point points[], int numPoints) {
    printf("Populating grid (%d cells)\n", gridWidth * gridHeight);
    // Populate the grid with points
    for(int i = 0; i < numPoints; i++) {
        Point point = points[i];

        int gridX = point.x / gridSize;
        int gridY = point.y / gridSize;
        int cellIndex = gridY * gridWidth + gridX;
        int newCount = cellPointCounts[cellIndex] + 1;

        // printf("Point: (%d, %d) index %d\n", point.x, point.y, i);
        // printf("Grid cell: (%d, %d) index %d\n", gridX, gridY, cellIndex);

        // printf("Cell index: %d\n", cellIndex);
        if(cellIndex >= gridWidth * gridHeight) {
            printf("!!!! Out of bounds\n");
            break;
        }

        if(newCount >= maxPointsPerCell * 0.8) {
            // printf("Point: (%d, %d) index %d\n", point.x, point.y, i);
            printf("!!!! Too many points in cell: %d\n", newCount);
            break;
        }
        if (newCount < maxPointsPerCell) {
            grid[cellIndex * maxPointsPerCell + newCount] = point;
            cellPointCounts[cellIndex] = newCount;
        }else {
            printf("!!!! Too many points in cell: %d\n", newCount);
        }
    }
    printf("Done populating grid\n");
}

__global__ void makeDepthMapKernel(Point* points, int numPoints, unsigned char* imageData, int imageWidth, int imageHeight, int gridSize, float distanceThreshold, int maxPointsPerCell, Point* grid, int* cellPointCounts) {
    // Compute the pixel coordinates for this thread
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image bounds
    if (pixelX >= imageWidth || pixelY >= imageHeight) {
        return;
    }

    int pixelIndex = pixelY * imageWidth + pixelX;

    // Get the local points for this pixel
    Point nearbyPoints[300];
    int numOfNearbyPoints = 0;

    int gridX = pixelX / gridSize;
    int gridY = pixelY / gridSize;

    for (int offsetX = -3; offsetX <= 3; offsetX++) {
        for (int offsetY = -3; offsetY <= 3; offsetY++) {
            int cX = gridX + offsetX;
            int cY = gridY + offsetY;

            if (cX >= 0 && cX < (imageWidth + gridSize - 1) / gridSize && cY >= 0 && cY < (imageHeight + gridSize - 1) / gridSize) {
                int cellIndex = cY * ((imageWidth + gridSize - 1) / gridSize) + cX;

                for (int k = 0; k < cellPointCounts[cellIndex]; ++k) {
                    const Point& point = grid[cellIndex * maxPointsPerCell + k];
                    float distance = sqrtf(powf(point.x - pixelX, 2) + powf(point.y - pixelY, 2));

                    if (distance <= distanceThreshold) {
                        nearbyPoints[numOfNearbyPoints] = point;
                        numOfNearbyPoints++;

                        if (numOfNearbyPoints >= 300) {
                            break;
                        }
                    }
                }
            }
        }
    }

    // Calculate the weights based on the Gaussian function
    float weights[300] = {0};
    float weightSum = 0;
    float sigma = distanceThreshold / 4;

    for (int j = 0; j < numOfNearbyPoints; j++) {
        float pointDistance = sqrtf(powf(nearbyPoints[j].x - pixelX, 2) + powf(nearbyPoints[j].y - pixelY, 2));
        weights[j] = expf(-powf(pointDistance, 2) / (2 * powf(sigma, 2)));
        weightSum += weights[j];
    }

    // Normalize the weights
    if (weightSum > 0) {
        for (int j = 0; j < numOfNearbyPoints; j++) {
            weights[j] /= weightSum;
        }
    }

    // Calculate the weighted sum of the z-values
    float value = 0;
    for (int j = 0; j < numOfNearbyPoints; j++) {
        value += nearbyPoints[j].z * weights[j];
    }

    // Scale the value to be between 0 and 255
    value = value * 255.0f / 255.0f;

    // Cap the value to be within the range [0, 255]
    if (value > 255) {
        value = 255;
    } else if (value < 0) {
        value = 0;
    }

    // Set the color of the pixel
    imageData[pixelIndex] = (unsigned char)value;
}

void DepthMap::makeDepthMap(Point points[], int numPoints, PPMImage* depthMapOutput) {
    // Allocate memory on the GPU
    Point* d_points;
    unsigned char* d_imageData;
    Point* d_grid;
    int* d_cellPointCounts;

    int gridCells = gridWidth * gridHeight;
    int imagePixels = imageWidth * imageHeight;

    populateGrid(points, numPoints);

    cudaMalloc(&d_points, numPoints * sizeof(Point));
    cudaMalloc(&d_imageData, imagePixels * sizeof(unsigned char));
    cudaMalloc(&d_grid, gridCells * maxPointsPerCell * sizeof(Point));
    cudaMalloc(&d_cellPointCounts, gridCells * sizeof(int));

    // Copy data to the GPU
    cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, grid, gridCells * maxPointsPerCell * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellPointCounts, cellPointCounts, gridCells * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((imageWidth + blockSize.x - 1) / blockSize.x, (imageHeight + blockSize.y - 1) / blockSize.y);

    makeDepthMapKernel<<<gridSize, blockSize>>>(d_points, numPoints, d_imageData, imageWidth, imageHeight, gridCells, distanceThreshold, maxPointsPerCell, d_grid, d_cellPointCounts);

    // Copy the result back to the host
    cudaMemcpy(imageData, d_imageData, imagePixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_points);
    cudaFree(d_imageData);
    cudaFree(d_grid);
    cudaFree(d_cellPointCounts);

    // Debug, to theck that there are poitns in the grid
    // for (int i = 0; i < gridCells; i++) {
    //     printf("Cell %d: %d points\n", i, cellPointCounts[i]);
    //     for (int j = 0; j < cellPointCounts[i]; j++) {
    //         printf("Point %d: (%d, %d, %d)\n", j, grid[i * maxPointsPerCell + j].x, grid[i * maxPointsPerCell + j].y, grid[i * maxPointsPerCell + j].z);
    //     }
    // }

    // Set the output image
    depthMapOutput->width = imageWidth;
    depthMapOutput->height = imageHeight;
    depthMapOutput->maxColor = 255;
    depthMapOutput->data = imageData;
}