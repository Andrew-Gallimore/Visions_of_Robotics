#include "makeDepthMap.h"

using namespace std;

// Makes a depth map from the points
// void makeDepthMap(vector<Point>& points, int imageWidth, int imageHeight) {
//     // Variables
//     int maxColor = 255;
//     int imagePixels = imageWidth * imageHeight;
//     unsigned char *imageData;
//     imageData = (unsigned char *)malloc(imagePixels * sizeof(unsigned char));

//     // Loop over all the pixels
//     for (int i = 0; i < imagePixels; i++) {
//         // Get the pixel coordinates
//         int pixelX = i % imageWidth;
//         int pixelY = i / imageWidth;

//         // Calculate the weights based on the inverse distance
//         vector<float> weights(points.size(), 0);
//         float weightSum = 0;
//         for (int j = 0; j < (int)points.size(); j++) {
//             float pointDistance = sqrt(pow(points[j].x - pixelX, 2) + pow(points[j].y - pixelY, 2));
//             if (pointDistance == 0) {
//                 weights[j] = 1;
//                 weightSum = 1;
//                 break;
//             } else {
//                 weights[j] = 1.0 / pointDistance;
//                 weightSum += weights[j];
//             }
//         }

//         // Normalize the weights so that their sum equals 1
//         for (int j = 0; j < (int)points.size(); j++) {
//             weights[j] /= weightSum;
//         }

//         // Calculate the weighted sum of the z-values
//         float value = 0;
//         for (int j = 0; j < (int)points.size(); j++) {
//             value += points[j].z * weights[j];
//         }

//         // Scale the value to be between 0 and 255
//         value = value * maxColor / 255.0;

//         // Cap the value to be within the range [0, 255]
//         if (value > 255) {
//             value = 255;
//         } else if (value < 0) {
//             value = 0;
//         }

//         // Set the color of the pixel
//         imageData[i] = (unsigned char)value;
//     }

//     // Write the image to a file
//     writePPM("depthMap.ppm", imageWidth, imageHeight, maxColor, 0, imageData);

//     // Free the allocated memory
//     free(imageData);
// }



void makeDepthMap(vector<Point>& points, int imageWidth, int imageHeight, int gridSize, float distanceThreshold) {
    // Variables
    int maxColor = 255;
    int imagePixels = imageWidth * imageHeight;
    unsigned char *imageData = (unsigned char *)malloc(imagePixels * sizeof(unsigned char));

    // Variables for debugging
    int averageLocalPoints = 0;
    int averageCheckedLocalPoints = 0;

    // Create a grid for spatial partitioning
    int gridWidth = (imageWidth + gridSize - 1) / gridSize;
    int gridHeight = (imageHeight + gridSize - 1) / gridSize;
    vector<vector<vector<Point>>> grid(gridWidth, vector<vector<Point>>(gridHeight));

    // Populate the grid with points
    for (const Point& point : points) {
        int gridX = point.x / gridSize;
        int gridY = point.y / gridSize;
        grid[gridX][gridY].push_back(point);
    }

    // Loop over all the pixels
    for (int i = 0; i < imagePixels; i++) {
        // Get the pixel coordinates
        int pixelX = i % imageWidth;
        int pixelY = i / imageWidth;

        // Determine the grid cell of the pixel
        int gridX = pixelX / gridSize;
        int gridY = pixelY / gridSize;

        // Collect points from neighboring grid cells within the distance threshold
        int numOfNearbyPoints = 0;
        Point nearbyPoints[100] = {0, 0, 0}; // Shouldn't ever exceed 100 points

        for (int OffsetX = -3; OffsetX <= 3; OffsetX++) {
            for (int OffsetY = -3; OffsetY <= 3; OffsetY++) {
                // Get a neighboring cell to grab points from
                int cX = gridX + OffsetX;
                int cY = gridY + OffsetY;

                // Checking if its withing the image
                if (cX >= 0 && cX < gridWidth && cY >= 0 && cY < gridHeight) {
                    for (const Point& point : grid[cX][cY]) {
                        float distance = sqrt(pow(point.x - pixelX, 2) + pow(point.y - pixelY, 2));
                        
                        averageCheckedLocalPoints++;

                        // Only adding the point if its within distance threshold
                        if (distance <= distanceThreshold) {
                            // NOTE: numNearbyPoints is still accumulating at this point
                            nearbyPoints[numOfNearbyPoints] = point;
                            numOfNearbyPoints++;
                        }
                    }
                }
            }
        }

        averageLocalPoints += numOfNearbyPoints;

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
    writePPM("depthMap.ppm", imageWidth, imageHeight, maxColor, 0, imageData);

    // Free the allocated memory
    free(imageData);
}

// void makeDepthMap(vector<Point>& points, int imageWidth, int imageHeight, int gridSize, float distanceThreshold) {
//     // Variables
//     int maxColor = 255;
//     int imagePixels = imageWidth * imageHeight;
//     unsigned char *imageData = (unsigned char *)malloc(imagePixels * sizeof(unsigned char));
//     memset(imageData, 0, imagePixels * sizeof(unsigned char));

//     // Variables for debugging
//     atomic<int> averageLocalPoints(0);
//     atomic<int> averageCheckedLocalPoints(0);

//     // Create a grid for spatial partitioning
//     int gridWidth = (imageWidth + gridSize - 1) / gridSize;
//     int gridHeight = (imageHeight + gridSize - 1) / gridSize;
//     vector<vector<vector<Point>>> grid(gridWidth, vector<vector<Point>>(gridHeight));

//     // Populate the grid with points
//     for (const Point& point : points) {
//         int gridX = point.x / gridSize;
//         int gridY = point.y / gridSize;
//         grid[gridX][gridY].push_back(point);
//     }

//     // Function to process a range of pixels
//     auto processPixels = [&](int start, int end) {
//         for (int i = start; i < end; i++) {
//             // Get the pixel coordinates
//             int pixelX = i % imageWidth;
//             int pixelY = i / imageWidth;

//             // Determine the grid cell of the pixel
//             int gridX = pixelX / gridSize;
//             int gridY = pixelY / gridSize;

//             // Collect points from neighboring grid cells within the distance threshold
//             int numOfNearbyPoints = 0;
//             Point nearbyPoints[100]; // Shouldn't ever exceed 100 points

//             for (int OffsetX = -3; OffsetX <= 3; OffsetX++) {
//                 for (int OffsetY = -3; OffsetY <= 3; OffsetY++) {
//                     // Get a neighboring cell to grab points from
//                     int cX = gridX + OffsetX;
//                     int cY = gridY + OffsetY;

//                     // Checking if its within the image
//                     if (cX >= 0 && cX < gridWidth && cY >= 0 && cY < gridHeight) {
//                         for (const Point& point : grid[cX][cY]) {
//                             float distance = sqrt(pow(point.x - pixelX, 2) + pow(point.y - pixelY, 2));

//                             averageCheckedLocalPoints++;

//                             // Only adding the point if its within distance threshold
//                             if (distance <= distanceThreshold) {
//                                 nearbyPoints[numOfNearbyPoints] = point;
//                                 numOfNearbyPoints++;
//                             }
//                         }
//                     }
//                 }
//             }

//             averageLocalPoints += numOfNearbyPoints;

//             // Calculate the weights based on the Gaussian function
//             float weights[100] = {0};
//             float weightSum = 0;
//             float sigma = distanceThreshold / 4;  // Standard deviation for Gaussian function
//             for (int j = 0; j < numOfNearbyPoints; j++) {
//                 float pointDistance = sqrt(pow(nearbyPoints[j].x - pixelX, 2) + pow(nearbyPoints[j].y - pixelY, 2));
//                 weights[j] = exp(-pow(pointDistance, 2) / (2 * pow(sigma, 2)));
//                 weightSum += weights[j];
//             }

//             // Normalize the weights so that their sum equals 1
//             if (weightSum > 0) {
//                 for (int j = 0; j < numOfNearbyPoints; j++) {
//                     weights[j] /= weightSum;
//                 }
//             }

//             // Calculate the weighted sum of the z-values
//             float value = 0;
//             for (int j = 0; j < numOfNearbyPoints; j++) {
//                 value += nearbyPoints[j].z * weights[j];
//             }

//             // Scale the value to be between 0 and 255
//             value = value * maxColor / 255.0;

//             // Cap the value to be within the range [0, 255]
//             if (value > 255) {
//                 value = 255;
//             } else if (value < 0) {
//                 value = 0;
//             }

//             // Set the color of the pixel
//             imageData[i] = (unsigned char)value;
//         }
//     };

//     // Determine the number of threads to use
//     int numThreads = 4;
//     vector<thread> threads;
//     int pixelsPerThread = imagePixels / numThreads;

//     // Launch threads to process pixels in parallel
//     for (int t = 0; t < numThreads; t++) {
//         int start = t * pixelsPerThread;
//         int end = (t == numThreads - 1) ? imagePixels : start + pixelsPerThread;
//         threads.emplace_back(processPixels, start, end);
//     }

//     // Wait for all threads to finish
//     for (thread& t : threads) {
//         t.join();
//     }

//     // Print the average number of local points
//     printf("Average number of local points: %d\n", averageLocalPoints / imagePixels);
//     printf("Average number of checked local points: %d\n", averageCheckedLocalPoints / imagePixels);

//     // Write the image to a file
//     writePPM("depthMap.ppm", imageWidth, imageHeight, maxColor, 0, imageData);

//     // Free the allocated memory
//     free(imageData);
// }