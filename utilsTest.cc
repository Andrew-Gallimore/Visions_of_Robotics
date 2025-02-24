// Homework 1, Robotics 444
// By: Andrew Gallimore

#include "utils/vectorUtils.h"
#include "utils/matrixUtils.h"
#include <stdio.h>

using namespace std;

int main () 
{
    float a[] = { 1, 2, 3 };
    float b[] = { 9, 1, 4 };
    float c[] = {-7, 3, 4 };

    int rows = 3;
    float scalar = 2.0;

    printf("====== Testing vector functions ======\n\n");
    
    // Print initial values
    printf("Printing vector a:\n");
    vectorPrint(a, rows);
    printf("Printing vector b:\n");
    vectorPrint(b, rows);
    printf("Printing vector c:\n");
    vectorPrint(c, rows);

    // Testing scaling
    float d[] = { 0, 0, 0 };
    printf ("Printing the vector a scaled by 2:\n");
    vectorScale(a, rows, scalar, d);
    vectorPrint(d, rows);

    // Testing magnitude
    printf("Printing the magnitude of vector a:\n");
    printf("\n");
    printf("    %.2f\n", vectorMag(a, rows));
    printf("\n");

    // Testing normalization
    printf("Printing vector a normalized:\n");
    vectorNorm(a, rows);
    vectorPrint(a, rows);

    printf("Printing the magnitude of vector a normalized (should be 1):\n\n    %.2f\n\n", vectorMag(a, rows));

    // Testing addition
    float e[] = {0,0,0};
    printf("Printing vector b and c added:\n");
    vectorAdd(e, b, c, rows);
    vectorPrint(e, rows);

    // Testing subtraction
    float f[] = {0,0,0};
    printf("Printing vector b and c subtracted:\n");
    vectorSub(f, b, c, rows); 
    vectorPrint(f, rows);

    // Testing dot product
    printf("Printing the dot of b on c:\n\n");
    printf("  %.2f\n", vectorDotProduct(b, c, rows));
    printf("\n");

    printf("====== Testing Matrix functions ======\n\n");

    float matrixA[] = { 1, 2, 3,
                        4, 5, 6,
                        7, 8, 9 };
    float matrixB[9] = {0};
    float matrixQ[9] = {0};
    float matrixR[9] = {0};

    int rowsA = 3;
    int colsA = 3;

    printf("Printing matrix A:\n");
    matrixPrint(matrixA, rowsA, colsA);

    printf("Printing matrix A transposed:\n");
    matrixTranspose(matrixA, rowsA, colsA, matrixB);
    matrixPrint(matrixB, colsA, rowsA);

    printf("Printing matrix Q and R of A QR decomposed :\n");
    matrixQR(matrixA, rowsA, colsA, matrixQ, matrixR);
    matrixPrint(matrixQ, rowsA, colsA);
    matrixPrint(matrixR, rowsA, colsA);

    printf("Printing the product of Q and R (should be A):\n");
    float matrixC[9] = {0};
    matrixProduct(matrixQ, rowsA, colsA, matrixR, rowsA, colsA, matrixC);
    matrixPrint(matrixC, rowsA, colsA);
    
}
