#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "utils/imageUtils.h"
#include "matchingCudaFunct.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

int fps = 10;

//Read rectification lookup tables
Mat map1x,map1y,map2x,map2y;
FileStorage fs("lookupTables.xml",FileStorage::READ);
fs["Map1x"]>>map1x;
fs["Map1y"]>>map1y;
fs["Map2x"]>>map2x;
fs["Map2y"]>>map2y;
fs.release();

if( map1x.empty()) cout << "Empty 1x lookup table"<<endl;
if( map1y.empty()) cout << "Empty 1y lookup table"<<endl;
if( map2x.empty()) cout << "Empty 2x lookup table"<<endl;
if( map2y.empty()) cout << "Empty 2y lookup table"<<endl;

string left_cam_pipeline  = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=" + to_string(fps) + "/1 ! nvvidconv flip-method=rotate-180 ! video/x-raw, format=GRAY8 !  appsink drop=1";
  
string right_cam_pipeline = "nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=" + to_string(fps) + "/1 ! nvvidconv flip-method=rotate-180 ! video/x-raw, format=GRAY8 !  appsink drop =1";

// Capture left/right frames
VideoCapture capL(left_cam_pipeline, CAP_GSTREAMER);
VideoCapture capR(right_cam_pipeline,CAP_GSTREAMER);

Mat left_m;
Mat right_m;

while(1) {
	capL >> left_m;
	capR >> right_m;

	// Apply rectification
        Mat rectifiedLeft, rectifiedRight, both;
        remap(left_m, rectifiedLeft, map1x, map1y, INTER_LINEAR);
        remap(right_m, rectifiedRight, map2x, map2y, INTER_LINEAR);

	// Turn mat's into ppm format
	PPMImage left = {
		capL.get(CAP_PROP_FRAME_WIDTH),
		capL.get(CAP_PROP_FRAME_HEIGHT),
		255,
		rectifiedLeft.data
	};

	PPMImage right = {
		capR.get(CAP_PROP_FRAME_WIDTH),
	        capR.get(CAP_PROP_FRAME_HEIGHT),
	        255,
	        rectifiedRight.data
	};  

	for(int i = (left.width * left.height) - 1; i >= 0; i--) {
		int newIndex = i + left.width * 21;
		if(newIndex < left.width * left.height) {
			right.data[newIndex] = right.data[i];
		}
	}
	PPMImage depthMap = {
                capR.get(CAP_PROP_FRAME_WIDTH),
                capR.get(CAP_PROP_FRAME_HEIGHT),
                255,
                rectifiedRight.data
        };


	// Do depth calculation
	matchingFunct(&left, &right, &depthMap);

	// Output them to ppm image
	//writePPM("testCapture.ppm", left.width, left.height, left.maxColor, 0, left.data);
	Mat img(depthMap.height, depthMap.width, CV_8UC1, depthMap.data);	

	imshow("depthImageVideo", img);
	
	hconcat(rectifiedLeft, rectifiedRight,both);
	imshow("Left and Right",both);

	waitKey(100);
}


// Release resources
capL.release();
capR.release();
destroyAllWindows();
  
return 0;
}
