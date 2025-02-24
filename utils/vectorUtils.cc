// Homework 1, Robotics 444
// By: Andrew Gallimore and Ben Kanter

#include <stdio.h>
#include <cmath>
#include "vectorUtils.h"

using namespace std;

void vectorPrint(float* u , int rows) {
  printf ("\n");
  for(int i=0;i<rows;i++) {
    printf( "%7.1f \n",u[i]);
  }
  printf("\n");
}

void vectorScale(float* u, int rows, float scalar, float* result) {
  for(int i=0; i<rows; i++) {
    result[i]=scalar*u[i];
  }
}

void vectorNorm(float* a, int rows) {
  // scale by 1 over magnitude to normalize
  vectorScale(a, rows, (1 / vectorMag(a, rows)), a);
}
float vectorMag(float* a, int rows) {
  // the sqrt of a dotted with itself is the magnitude
  return sqrt(vectorDotProduct(a, a, rows));
}

void vectorAdd(float* result, float* a, float* b, int rows) {
  // Adding each element in the vectors
  for (int c = 0; c < rows; c++) {
    result[c] = a[c] + b[c];
  }
}

void vectorSub(float* result, float* a, float* b, int rows) {
  // Subtracting each element in the vectors
  for (int c = 0; c < rows; c++) {
    result[c] = a[c] - b[c];
  }
}


float vectorDotProduct(float* a, float* b, int rows) {
  float dot = 0.0;

  // Calculating the dot product by multiplying each element in the vectors, then suming
  for (int c = 0; c < rows; c++) {
    dot += (a[c] * b[c]);
  }

  // And returning it
  return dot;
}

