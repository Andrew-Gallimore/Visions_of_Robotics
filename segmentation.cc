#include "segmentation.h"
using namespace std;

struct Cluster {
    Point center;
    vector<Point> points;
};


void findNearbyCenters(Cluster clusters[], int numClusters, int nearbyCenterIndexes[], int& numNearbyCenters) {
    // TEMPORARY
    numNearbyCenters = numClusters;
    for(int i = 0; i < numClusters; i++) {
        nearbyCenterIndexes[i] = i;
    }
    // If numNearbyCenters >= numClusters, 
        // Set numNearbyCenters
        // return centers indexes as nearbyCenters

    // While number of centers isn't reached
        // Add another grid-cell to the search
            // NOTE: Should add in a spiral pattern, starting from the corner/side the center is on in the first cell

        // Loop over all the centers in new grid-cell
            // If the center is within a certain distance, add it to the nearbyCenters array
            // Increment the numNearbyCenters
}

void segmentImage(PPMImage* image, int imageWidth, int imageHeight, Point centers[], int numClusters, int itterations) {
    // 0 follows color contures better, up to 40 or higher makes each segment more square.
    const float compactness = 30.0;
    const float step = sqrt((imageWidth * imageHeight) / numClusters);

    // Place to temporaroly store the clusters found
    Cluster clusters[numClusters];

    // Initialize clusters in a grid pattern
    int index = 0;
    for (int y = step / 2; y < imageHeight; y += step) {
        for (int x = step / 2; x < imageWidth; x += step) {
            if (index < numClusters) {
                int pixelColor = image->data[y * imageWidth + x];
                clusters[index].center = {x, y, pixelColor};
                index++;
            }
        }
    }

    // Loop for number of itterations
    for(int i = 0; i < itterations; i++) {
        printf("Itteration: %d\n", i);
        // Clear clusters
        for (int i = 0; i < numClusters; ++i) {
            clusters[i].points.clear();
        }

        // Loop over all the pixels
        for(int j = 0; j < imageWidth * imageHeight; j++) {
            // printf("Pixel: %d\n", j);
            // Get point
            Point pixel;
            pixel.y = j / imageWidth;
            pixel.x = j % imageWidth;
            pixel.z = image->data[j];

            // Find nearby centers
            int numNearbyCenters = 8;
            int nearbyCenterIndexes[numNearbyCenters];
            findNearbyCenters(clusters, numClusters, nearbyCenterIndexes, numNearbyCenters);
            
            // Find closest center
            int closestDistance = 1000000;
            int closestIndex = -1;
            for(int k = 0; k < numNearbyCenters; k++) {
                Cluster* clust = &clusters[nearbyCenterIndexes[k]];
                
                // Calculating a best place for it to go, based on spacial and color distances
                float distanceColor = abs(pixel.z - clust->center.z);
                float distanceSpace = sqrt((pixel.x - clust->center.x) * (pixel.x - clust->center.x) + (pixel.y - clust->center.y) * (pixel.y - clust->center.y));
                
                float distance = sqrt(distanceColor * distanceColor + (distanceSpace / step) * (distanceSpace / step) * compactness * compactness);
                
                // If closer, set a closer cluster/center
                if(distance < closestDistance) {
                    closestIndex = nearbyCenterIndexes[k];
                    closestDistance = distance;
                }
            }

            // Put the pixel in that nearest center's cluster
            clusters[closestIndex].points.push_back(pixel);
        }
        
        printf("Done with itteration: %d\n", i);

        // Calculate the new center of each cluster
        // For every cluster
        for(int j = 0; j < numClusters; j++) {
            printf("Cluster: %d\n", j);
            int numPoints = clusters[j].points.size();
            int xSum = 0;
            int ySum = 0;
            
            // For every pixel in that cluster
            for(int k = 0; k < numPoints; k++) {
                printf("Point: %d\n", k);
                xSum += clusters[j].points[k].x;
                ySum += clusters[j].points[k].y;
            }
            printf("HERE: %d\n", j);
            
            // Setting cluster center to new average
            clusters[j].center.x = xSum / numPoints;
            clusters[j].center.y = ySum / numPoints;
        }

        printf("Finished setting new centers: %d\n", i);
    }

    //printf("Done with all itterations\n");

    // Color pixels to some proxy of their ID
    for(int i = 0; i < numClusters; i++) {
        //printf("Cluster: %d\n", i);
        int numPoints = clusters[i].points.size();
        int color = rand() * 255;

        // For every pixel in that cluster
        for(int j = 0; j < numPoints; j++) {
            // Get the pixel index
            int pixelIndex = (clusters[i].points[j].y * imageWidth) + (clusters[i].points[j].x);
            //printf("Point: %d, (%d, %d) index: %d\n", j, clusters[i].points[j].x, clusters[i].points[j].y, pixelIndex);

            // Set pixel to be the random color for this cluster
            image->data[pixelIndex] = color;
        }
    }

    //printf("Done coloring the image\n");
}

