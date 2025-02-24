#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <string>
#include <sstream>
#include "calibFile.h"
#include "utils/matrixUtils.h"
#include "utils/vectorUtils.h"

using namespace std;

int main() {
   
   // Getting name of file
   cout << "Enter the name of the calibration file: ";
   string inputFile;
   getline(std::cin, inputFile); // Read file name from user

   // Instantiate the calibFile class and read data
   calibFile calib(inputFile);

   // Get the number of points
   int numPoints = calib.getNumPoints();
   printf("Number of Points %d \n", numPoints);

   // Early return if not enough points
   if (numPoints < 12) {
       printf("Not enough points. Min of 12 are needed \n");
       exit(0);
   }

   // Define input vectors using getters
   const float* u = calib.getU();
   const float* v = calib.getV();
   const float* x = calib.getX();
   const float* y = calib.getY();
   const float* z = calib.getZ();
   const float* zc = calib.getZc();
   
   // The size of the M and M-transpose matrix's
   int rows = 3 * numPoints;
   int cols = 12;
   int Msize = rows * cols;


   // ==== Making M Matrix ====
   float M[Msize];
   int arr_counter = 0;
   int ptr_tracker = -1;

   // Filling in M
   for(int i = 0; i < rows; i++) {
      if (i % 3 == 0) ptr_tracker++;

      for (int j = 0; j < cols; j++) {
         if (i % 3 == 0) {
            if (j < 4) {
               if (arr_counter % 4 == 0) { M [arr_counter] = x[ptr_tracker]; }
               else if (arr_counter % 4 == 3) { M [arr_counter] = 1; }
               else if (arr_counter % 4 == 2) { M [arr_counter] = z[ptr_tracker]; }
               else if (arr_counter % 4 == 1) { M [arr_counter] = y[ptr_tracker]; }
            } else { M [arr_counter] = 0; }
         }
         else if (i % 3 == 1) {
            if (j >= 4 && j < 8) {
               if (arr_counter % 4 == 0) { M [arr_counter] = x[ptr_tracker]; }
               else if (arr_counter % 4 == 3) { M [arr_counter] = 1; }
               else if (arr_counter % 4 == 2) { M [arr_counter] = z[ptr_tracker]; }
               else if (arr_counter % 4 == 1) { M [arr_counter] = y[ptr_tracker]; }
            } else { M [arr_counter] = 0; }
         }
         else if (i % 3 == 2) {
            if (j >= 8) {
               if (arr_counter % 4 == 0) { M [arr_counter] = x[ptr_tracker]; }
               else if (arr_counter % 4 == 3) { M [arr_counter] = 1; }
               else if (arr_counter % 4 == 2) { M [arr_counter] = z[ptr_tracker]; }
               else if (arr_counter % 4 == 1) { M [arr_counter] = y[ptr_tracker]; }
            } else { M [arr_counter] = 0; }
         }

         arr_counter++;
      }
   }


   // ==== Making M Transpose ====
   float Mtrans[Msize];

   matrixPrint (M, rows, cols);
   matrixTranspose (M, rows, cols, Mtrans);
   matrixPrint (Mtrans, cols, rows);


   // ==== Making A Matrix ====
   float A[12 * 12] = {0};
   
   // Using equation Mt * M = A 
   matrixProduct (Mtrans, cols, rows, M, rows, cols, A);
   printf ("A MATRIX\n");
   matrixPrint (A, cols, cols);
   

   // ==== QR Decomposition of A ====
   float Q[cols * cols] = {0};
   float R[cols * cols] = {0};
   
   matrixQR (A, cols, cols, Q, R);
   printf ("Q MATRIX\n");
   matrixPrint (Q, cols, cols);
   printf ("R MATRIX\n");
   matrixPrint(R, cols, cols);


   // ==== Making Qt ====
   float Qt[cols * cols] = {0};
   matrixTranspose (Q, cols, cols, Qt);
   

   // ==== Verifying QR Decomposition is working ====
   float Qi[cols * cols] = {0};

   // Using equation Q * Qt = I to verify an identity is formed
   matrixProduct (Q, cols, cols, Qt, cols, cols, Qi);
   matrixPrint (Qi, cols, cols);


   // ==== Making vector c ====
   int row_b = 0;
   float c[rows] = {0};
   
   // Filling in vector c
   for (int i = 0; i < numPoints; i++) {
      c [row_b] = (u [i] * zc [i]);
      row_b++;
      c [row_b] = (v [i] * zc [i]);
      row_b++;
      c [row_b] = (zc [i]);
      row_b++;
   }
   

   // ==== Making vector b ====
   float b[cols] = {0};

   // Using equation b = Mt * c
   matrixTimesVector (Mtrans, cols, rows, c, rows, b);
   printf ("b = mT*c should be \n");
   vectorPrint (b, cols);
   

   // ==== Making vector d ====
   float d[cols] = {0};

   // Using equation d = Qt * b
   matrixTimesVector (Qt, cols, cols, b, cols, d);


   // ==== Back substitution ==== 
   float p[cols] = {0};
   matrixBackSubstitution (R, cols, cols, d, p);
   printf ("Projection Matrix \n");
   matrixPrint (p, 3, 4);   


   // ==== Calibration Matrix k ====
   float k[9] = {0};
   matrixInternalCameraParameters (p, 3, 4, k);
   printf ("CALIB MATRIX \n");
   matrixPrint (k, 3, 3);

   return 0;
}
