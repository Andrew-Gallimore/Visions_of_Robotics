#include "utils/imageUtils.h"
#include "utils/timer.h"
#include "Structs.h"
#include "segmentation.h"

int main() {
    // Read in the image
    convertJPGToPPM("images/newLeft.jpg", "images/newLeft.ppm");
    convertPPMToBW("images/newLeft.ppm", "images/newLeftBW.ppm");
    PPMImage* image = readPPM("images/newLeftBW.ppm", 0);

    // Run segmentation
    int numClusters = 15;
    int itterations = 1;
    Point centers[numClusters];
    printf("Segmenting image\n");
    segmentImage(image, image->width, image->height, centers, numClusters, itterations);

    // make ppm with the pixels
    writePPM("segmented.ppm", image->width, image->height, 255, 0, image->data);

    return 0;
}