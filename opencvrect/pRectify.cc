#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    
// Load left and right stereo images
    Mat leftImageColor  = imread("nvcamtest_11114_s00_00000.jpg");
    Mat rightImageColor = imread("nvcamtest_11219_s01_00000.jpg");

   // printf ("HERE\n"); 

    cv::Mat leftImage,rightImage;
    cv::cvtColor(leftImageColor,leftImage, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightImageColor,rightImage, cv::COLOR_BGR2GRAY);

    printf ("ALSO HERE\n");

    imshow("left",leftImage);
    imshow("right",rightImage);

    if (leftImage.empty() || rightImage.empty()) {
        cout << "Error: Could not load stereo images!" << endl;
        return -1;
    }

 
// Camera parameters (intrinsic matrices)
       Mat cameraMatrix1 = (Mat_<double>(3,3) << 566.5176, 0, 252.2899,
                                                 0, 753.3832, 218.9708,
                                                  0,0,1);
        Mat cameraMatrix2 = (Mat_<double>(3,3) << 542.131, 0, 256.252,
                                                  0, 720.9599, 274.971,
                                                  0 ,0, 1);
  
        // Distortion coefficients
        Mat distCoeffs1 = (Mat_<double>(1, 5) << -0.153454, 0.4751299, -0.009426, -0.013865, -0.5498623);
        Mat distCoeffs2 = (Mat_<double>(1, 5) << -0.0417512, 0.163885, -0.0069323, -0.0222308, -0.1230468);

    //Rotation matrix R:
    Mat R = (Mat_<double>(3,3) << 0.9996647904096437, 0.001965333417136132, -0.02581558211240997,
                              -0.002060580359348861, 0.9999911666058071, -0.003663429941946584,
                               0.02580815421191061, 0.003715397006562428, 0.9996600097039283);
    //Translation vector T:
    Mat T = (Mat_<double>(3,1) << -58.4741852215618,
                              0.5308936848201911,
                              1.695405159740024);
    // Output rectification transforms, projection matrices, and disparity-to-depth mapping matrix
    Mat R1, R2, P1, P2, Q;
    
    // Compute rectification transforms
    stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, leftImage.size(), 
                  R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY,0,leftImage.size());
    
    // Compute undistortion and rectification maps
    Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, leftImage.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, rightImage.size(), CV_32FC1, map2x, map2y);
    
    // Apply rectification
    Mat rectifiedLeft, rectifiedRight;
    
/*
    for (int i = (leftImage.size().width * leftImage.size().height) - 1; i >= 0; i--)
    {
        int newIndex = i + (leftImage.size().width * 100);
	if (newIndex < (leftImage.cols * leftImage.rows))
	{
		map2y.data[newIndex] = map2y.data[i];
	}
        else
	{
        	map2y.data[i] = 0;
	}
    }
*/
    //for(int i = 0; i < leftImage.size().width * leftImage.size().height; i++) {
//	map2y.data[i] += 21;
  //  }

    remap(leftImage, rectifiedLeft, map1x, map1y, INTER_LINEAR);
    remap(rightImage, rectifiedRight, map2x, map2y, INTER_LINEAR);
    

    FileStorage fs("../lookupTables.xml",FileStorage::WRITE);
    fs << "Map1x" << map1x;
    fs << "Map1y" << map1y;
    fs << "Map2x" << map2x;
    fs << "Map2y" << map2y;
    fs.release();

    // Display results
    imshow("Rectified Left Image", rectifiedLeft);
    imshow("Rectified Right Image", rectifiedRight);

   // Display original images
    imwrite("leftRectified.jpg", rectifiedLeft); 
    imwrite("rightRectified.jpg", rectifiedRight); 
    waitKey(0);
    return 0;
}

