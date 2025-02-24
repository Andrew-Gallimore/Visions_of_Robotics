#ifndef UTILS_H
#define UTILS_H

using namespace std;

void vectorPrint(float* u, int rows); 

void vectorScale(float* u, int rows, float alpha, float* v); 
void vectorSub(float*, float*, float*, int);
void vectorAdd(float*, float*, float*, int);
void vectorNorm(float*, int);
float vectorMag(float*, int);
float vectorDotProduct(float*, float*, int);

#endif

