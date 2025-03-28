#include "makeDepthMap.h"

using namespace std;

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

// void DepthMap::populateGrid(Point points[]) {
//     printf("Populating grid\n");
//     // Populate the grid with points
//     for(int i = 0; i < gridWidth * gridHeight; i++) {
//         Point point = points[i];

//         int gridX = point.x / gridSize;
//         int gridY = point.y / gridSize;
//         int cellIndex = gridY * gridWidth + gridX;
//         int newCount = cellPointCounts[cellIndex] + 1;

//         if(newCount >= maxPointsPerCell * 0.8) {
//             // printf("Point: (%d, %d) index %d\n", point.x, point.y, i);
//             printf("!!!! Too many points in cell: %d\n", newCount);
//             break;
//         }
//         if (newCount < maxPointsPerCell) {
//             grid[cellIndex * maxPointsPerCell + newCount] = point;
//             cellPointCounts[cellIndex] = newCount;
//         }
//     }
//     printf("Done populating grid\n");
// }

void DepthMap::populateGrid(Point points[], int numPoints) {
    printf("Populating grid (%d cells)\n", gridWidth * gridHeight);
    // Populate the grid with points
    for(int i = 0; i < numPoints; i++) {
        Point point = points[i];

        int gridX = point.x / gridSize;
        int gridY = point.y / gridSize;
        int cellIndex = gridY * gridWidth + gridX;
        int newCount = cellPointCounts[cellIndex] + 1;

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

void DepthMap::getLocalPoints(int pixelX, int pixelY, Point nearbyPoints[], int& numOfNearbyPoints) {
    // Determine the grid cell of the pixel
    int gridX = pixelX / gridSize;
    int gridY = pixelY / gridSize;
    
    // Getting local points by grabbing points in local cells
    for (int OffsetX = -3; OffsetX <= 3; OffsetX++) {
        for (int OffsetY = -3; OffsetY <= 3; OffsetY++) {
            // Get a neighboring cell to grab points from
            int cX = gridX + OffsetX;
            int cY = gridY + OffsetY;

            // Checking if it's within the image
            if (cX >= 0 && cX < gridWidth && cY >= 0 && cY < gridHeight) {
                int cellIndex = cY * gridWidth + cX;
                for (int k = 0; k < cellPointCounts[cellIndex]; ++k) {
                    const Point& point = grid[cellIndex * maxPointsPerCell + k];
                    float distance = sqrt(pow(point.x - pixelX, 2) + pow(point.y - pixelY, 2));
                    
                    // For debugging
                    pointsChecked++;

                    // Only adding the point if it's within distance threshold
                    if (distance <= distanceThreshold) {
                        // NOTE: numNearbyPoints is still accumulating at this point
                        nearbyPoints[numOfNearbyPoints] = point;
                        numOfNearbyPoints++;
                    }

                    if(numOfNearbyPoints >= 280) {
                        printf("!!!! Too many points in local area\n");
                        return;
                    }
                }
            }
        }
    }
}

void DepthMap::makeDepthMap(Point points[], int numPoints, PPMImage* depthMapOutput) {
    // Variables
    int maxColor = 255;
    int imagePixels = imageWidth * imageHeight;
    unsigned char *imageData = (unsigned char *)malloc(imagePixels * sizeof(unsigned char));
    
    int averageLocalPoints = 0;
    int averageCheckedLocalPoints = 0;
    printf("Here\n");
    
    // First, populate the grid with the points
    populateGrid(points, numPoints);

    printf("Here2\n");
    
    // Now, loop over all the pixels
    for (int i = 0; i < imagePixels; i++) {
        pointsChecked = 0; // Reset for each pixel
        
        // Get the pixel coordinates
        int pixelX = i % imageWidth;
        int pixelY = i / imageWidth;
        
        // Get the local points to pixel
        Point nearbyPoints[300] = {0}; // Gets populated by getLocalPoints
        int numOfNearbyPoints = 0;     // Gets set by getLocalPoints
        DepthMap::getLocalPoints(pixelX, pixelY, nearbyPoints, numOfNearbyPoints);

        // For debugging
        averageLocalPoints += numOfNearbyPoints;
        averageCheckedLocalPoints += pointsChecked;

        // Calculate the weights based on the Gaussian function
        float weights[numOfNearbyPoints] = {0};
        float weightSum = 0;
        float sigma = distanceThreshold / 4;  // Standard deviation for Gaussian function
        for (int j = 0; j < numOfNearbyPoints; j++) {
            float pointDistance = sqrt(pow(nearbyPoints[j].x - pixelX, 2) + pow(nearbyPoints[j].y - pixelY, 2));
            weights[j] = exp(-pow(pointDistance, 2) / (2 * pow(sigma, 2)));
            weightSum += weights[j];
        }

        // Normalize the weights so that their sum equals 1
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
        value = value * maxColor / 255.0;

        // Cap the value to be within the range [0, 255]
        if (value > 255) {
            value = 255;
        } else if (value < 0) {
            value = 0;
        }

        // Set the color of the pixel
        imageData[i] = (unsigned char)value;
    }

    // Print the average number of local points
    printf("Average number of local points: %d\n", averageLocalPoints / imagePixels);
    printf("Average number of checked local points: %d\n", averageCheckedLocalPoints / imagePixels);

    // Write the image to a file
    //writePPM("depthMap.ppm", imageWidth, imageHeight, maxColor, 0, imageData);
    *depthMapOutput = {
	imageWidth,
	imageHeight,
	maxColor,
	imageData
    };

    // Free the allocated memory
    free(imageData);
}
