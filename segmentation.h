#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <cstdio>
#include "Structs.h"
#include "cmath"
#include "utils/imageUtils.h"
#include <vector>

void segmentImage(PPMImage* image, int imageWidth, int imageHeight, Point centers[], int numClusters, int itterations);

#endif //SEGMENTATION_H