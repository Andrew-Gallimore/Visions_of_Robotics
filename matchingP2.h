#ifndef MATCHINGP2_H
#define MATCHINGP2_H

#include <cuda_runtime.h>
#include "utils/imageUtils.h"
#include "Structs.h"

__global__ void searchForPoints(Point* initialPoints, Point* matchPoints, int numPoints, int imageScalar, int windowSize, unsigned char* leftImage, unsigned char* rightImage, int imageWidth, int imageHeight);

#endif // MATCHINGP2_H