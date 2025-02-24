#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <string>
#include <sstream>
#include "vectorUtils.h"
#include "matrixUtils.h"

using namespace std;

void matrixPrint(float* matrix, int rows, int cols) {
    printf("number of rows and cols,%d %d \n", rows, cols);
    for (int row = 0; row < rows; row++) {

        for (int col = 0; col < cols; col++) {
            printf("%6.5f ",matrix[row*cols + col]);
        }
        printf("\n");

    }
    printf("\n");
}

void matrixProduct(float* a, int rows_a, int cols_a, float* b, int rows_b, int cols_b, float* c){
 int index = 0;
 int rows  = rows_a;
 int cols  = cols_b;

 if(cols_a == rows_b){
 for(int row = 0; row < rows; row++){
   for(int col = 0; col < cols; col++){
       index = row*cols + col;
       c[index] = 0.0; 
       for(int k = 0; k < cols_a; k++){
         c[index]=c[index]+a[row*cols_a + k]*b[col+k*cols_b];
       }
   }

 }

 }else{
	 printf("Can't multiply. Dimensions incorrect \n");
	 exit(0);
 }

}

void matrixTimesVector(float* a, int rows, int cols, float* v, int rows_v, float* w){
if(cols==rows_v){
   for( int row = 0; row < rows; row++){
                w[row]=0.0;
		for( int col = 0;col < cols;col++){
                   w[row]=w[row]+a[row*cols+col]*v[col];
		}
   } 
}else{
	printf(" dimensions don't match \n");
	exit(0);
}

}

void matrixTranspose(float* a, int rows, int cols, float* aTranspose){
 for( int row = 0; row < rows; row++){
		for( int col = 0;col < cols;col++){

		    aTranspose[col*rows+row] = a[row*cols+col];
		}
		 
 }
}

void matrixExtractCol(float*a, int rows, int cols, int col, float* column){
   int index;
    for(int row=0;row<rows;row++){
	index = col + row*cols;
        column[row] = a[index];
    }
}


void matrixBackSubstitution(float* R, int rows, int cols, float* d, float* p){
//solve Rp=d for an uppertriangular matrix R
// using back substitution

  for (int j = cols-1; j > -1; j--){
        p[j]=d[j];
	for(int k = j+1; k < cols; k++){
		p[j]= p[j] - R[j*cols + k]*p[k];
	}
	p[j]=p[j]/R[j*cols+j];
  }

}

void matrixUpperTriangularInverse(float* A, int rows, int cols, float* invA){
// This code computes the inverse of a non-singular upper triangular 3x3 matrix.
if(rows !=3 || cols !=3){
	printf(" Only 3x3 full rank upper triangle matricies can be inverted with this code\n");
       	exit(0);
}

 for(int row = 0; row < rows; row++){
  for(int col= 0;col < cols; col++){
    invA[ row*cols + col]  = 0.0;
  }
 }

invA[0*cols+0]=1.0/A[0*cols+0];
invA[1*cols+1]=1.0/A[1*cols+1];
invA[2*cols+2]=1.0/A[2*cols+2];

invA[0*cols+1]=-A[0*cols+1]*invA[1*cols+1]/A[0*cols+0];
invA[1*cols+2]=-A[1*cols+2]*invA[2*cols+2]/A[1*cols+1];
invA[0*cols+2]=-( A[0*cols+1]*invA[1*cols+2] + A[0*cols+2]*invA[2*cols+2] )/A[0*cols+0];
}

void matrixInternalCameraParameters(float* p,int rows,int cols,float* k){

float pSub[9];
float r[9];
float rInv[9];
float q[9];
float qT[9];
float b[9];

// extract a 3x3 sub matrix of p
for (int row=0; row < rows; row++){
	for (int col=0; col<cols-1;col++){
            pSub[row*(cols-1)+col]=p[row*cols+col];
	}
}
printf("3x3 submatrix of p\n");
matrixPrint(pSub,3,3);

// find the inverse of pSub with QR decomposition
// Its inverse is B=RInv*QTranspose
matrixQR(pSub,3,3,q,r);
matrixTranspose(q,3,3,qT);
matrixUpperTriangularInverse(r,3,3,rInv);
matrixProduct(rInv,3,3,qT,3,3,b);

//QR decompose the matrix B
matrixQR(b,3,3,q,r);

int rows_k = rows;
int cols_k = rows_k;

int rows_r = rows;
int cols_r = rows_r;

//diagonals of calibration matirx
k[0*cols_k +0] = 1/r[0*cols_r+0];
k[1*cols_k +1] = 1/r[1*cols_r+1];
k[2*cols_k +2] = 1.0;
// off diagonals of the calibration matrix
k[0*cols_k +2] = -r[0*cols_r +2]/r[0*cols_r+0];
k[1*cols_k +2] = -r[1*cols_r +2]/r[1*cols_r+1];



}

void matrixQR(float* a, int rows, int cols, float* Q, float* R) {
    // A & Q & R should be square matrices (rows x rows)
    // R is going to be upper triangular
    // Q is going to be orthogonal
        // Q's columns will also be orthonormal vectors

    // First, we need to make sure the matrix's are square
    if(rows != cols) {
        printf("Matrix is not square. Can't perform QR decomposition\n");
        return;
    }

    // Now, we need to get the q vectors by normalizing the columns of a

    for(int i = 0; i < cols; i++) {
        // Looking at the ith column of a
        float a_i_perpendicular[rows];

        // Start off a_i_perpendicular as a_i
        matrixExtractCol(a, rows, cols, i, a_i_perpendicular);

        // Calculating a_i_perpendicular (part of q_i vector)
        // Filling parts of R in the ith column while we're at it
        for(int j = 0; j < i; j++) {
            // Extracting the ith column of a
            float a_i[rows];
            matrixExtractCol(a, rows, cols, i, a_i);

            // Extracting the jth column of Q
            float q_j[rows];
            matrixExtractCol(Q, rows, cols, j, q_j);

            // Getting distance of a_i projected onto q_j
            float dotProduct = vectorDotProduct(a_i, q_j, rows);

            // Placing dotProduct into R (specifically located at r_ij)
            R[j*cols + i] = dotProduct;

            // Scaling q_j by dotProduct
            vectorScale(q_j, rows, dotProduct, q_j);

            // Subtracting the scaled q_j from a_i_perpendicular to make a new a_i_perpendicular
            vectorSub(a_i_perpendicular, a_i_perpendicular, q_j, rows);
        }

        // Now we have a complete a_i_perpendicular vector

        // Filling edge values of R in the ith column
        for(int j = 0; j < rows; j++) {
            // Looking at the jth row of the ith column of a
            if(j == i) {
                // Fill with the magnitude of a_i_perpendicular
                R[j*cols + i] = vectorMag(a_i_perpendicular, rows);
            }
        }

        // Now we fill the ith row of Q with the normalized a_i_perpendicular values
        vectorNorm(a_i_perpendicular, rows);
        for(int j = 0; j < rows; j++) {
            Q[j*cols + i] = a_i_perpendicular[j];
        }
    }

    // Now, we need to get the r values by getting the perpendicualr components of a onto the q vectors
        // q_i = a_i - a projected onto q_i direction - a projected onto q_i+1 direction - ..., all divided by the magnitude of all above
}
