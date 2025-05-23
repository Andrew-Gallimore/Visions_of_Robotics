#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "utils/imageUtils.h"
#include "matchingCudaFunct.h"
#include "serialUtils.h"
//#include <opencv1/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <unistd.h>

using namespace cv;
using namespace std;


// bool shouldStop (PPMImage *img)
// {
// 	int sum = 0;

// 	for (int i = 0; i < (img->width * img->height); i++ )
// 	{
// 		sum += img->data [i];


// 	}

// 	// cout << sum << endl;
// 	if (sum > (4000000))
// 	{
// 		return true;
// 	}


// 	return false;
// }

int startBuffer = 6;
char* previousMove = "";
char* previousSteer = "";

void handleObstacles (PPMImage *depthImg, int portID)
{
	int zones = 5;
	int zoneWidth = (depthImg->width)/zones;
	int zoneSums [zones] = {0};
	bool zonesBlocked [zones] = {false};

	char* mvCmd = "FWD090\n";
	char* strCmd = "STR100\n";

	// cout << "CURRENT PORT ID: " << portID;

	for (int i = 0; i < zones; i++)
	{
		for (int j = 0; j < zoneWidth; j++) // add sum for each width segment 

		{
			for (int h = 0; h < (depthImg->height); h++)
			{
					zoneSums [i] += depthImg->data [(h * depthImg->width) + (zoneWidth * i) + j];	
			}
		}
	}

	for (int i = 0; i < zones; i++)
	{
		cout << "Zone" << (i) << ":" << zoneSums [i] << ' ';

	    if (zoneSums [i] >= (2000000 / zones))
		{
			zonesBlocked [i] = true;
		}

	    cout << "Zone " << (i) << ":" << zonesBlocked [i] << endl;
	}
	cout << endl;

	bool mvOrStr = false; // TRUE - MOVE | FALSE - STR

	if ( zonesBlocked[0] && 
		 zonesBlocked[1] && 
		 zonesBlocked[2] && 
		 zonesBlocked[3] && 
		 zonesBlocked[4] ) // Total blockage : Reverse at spd 180
	{
		// mvCmd = "STP000\n";

		mvCmd = "REV070\n";
	}
	else if ( zonesBlocked[0] && 
		      zonesBlocked[1] && 
		      zonesBlocked[2] ) // left blockage : Steer right 40 degrees
	{ strCmd = "STR130\n"; }
	else if ( zonesBlocked[2] && 
		      zonesBlocked[3] && 
		      zonesBlocked[4] ) // right blockage : Steer left 40 degrees
	{ strCmd = "STR050\n"; }
	else if ( zonesBlocked[1] && 
		      zonesBlocked[2] && 
		      zonesBlocked[3] ) // middle blockage : Reverse at spd 110
	{
		mvCmd = "REV070\n";
	}
	else if ( zonesBlocked[0] ) // zone 0 blocked : Steer right 20 degrees
	{
		strCmd = "STR110\n";
	}
	else if ( zonesBlocked[4] ) // zone 4 blocked : Steer left 20 degrees 
	{
		strCmd = "STR070\n";
	}

	if(mvCmd != previousMove)
	{
		int mvBytesWritten = serialPortWrite (mvCmd, portID);

		if ( mvBytesWritten > 0 )
		{
			printf ("Sent %d bytes: %s\n", mvBytesWritten, mvCmd);

			if(startBuffer <= 0)
			{
				previousMove = mvCmd;
			}
			else
			{
				startBuffer--;
			}
		}
	}

	if(strCmd != previousSteer)
	{
		int strBytesWritten = serialPortWrite (strCmd, portID);

		if ( strBytesWritten > 0 )
		{
			printf ("Sent %d bytes: %s\n", strBytesWritten, strCmd);

			if(startBuffer <= 0)
			{
				previousSteer = strCmd;
			}
			else
			{
				startBuffer--;
			}
		}
	}
	
}

bool keys [256] = {0}; 

int main(int argc, char** argv) 
{

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

	/* capL >> left_m;
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
		int newIndex = i + left.width * 12;
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
	Mat img(depthMap.height, depthMap.width, CV_8UC1, depthMap.data); */

	int portID;
	portID = serialPortOpen ();

	if ( portID < 0 )
	{
		printf ("Error opening serial port \n");
		exit (0);
	}

	thread ([] () -> void {
							while (true) keys[(unsigned char) (cin.get ())] = true;
						  }).detach ();

	float offset = -9;
	float currentRow;
	for(int row = 0; row < map2y.rows; row++)
	{
		for(int col = 0; col < map2y.cols; col++)
		{
			currentRow = map2y.at<float>(row, col);
			if(currentRow + offset < 0 || currentRow + offset > map2y.rows) map2y.at<float>(row, col) = currentRow;
			else map2y.at<float>(row, col) = currentRow + offset;
		}
	}

	// for(int i = (map2y.cols * map2y.rows) - 1; i >= 0; i--) 
	// {
	// 		int newIndex = i + map2y.cols * 12;
	// 		if(newIndex < map2y.cols * map2y.rows) {
	// 			map2y.at<float>(newIndex) = map2y.at<float>(i);
	// 		}
	// 	}

	while ( 1 ) 
	{
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

		/* for(int i = (left.width * left.height) - 1; i >= 0; i--) {
			int newIndex = i + left.width * 12;
			if(newIndex < left.width * left.height) {
				right.data[newIndex] = right.data[i];
			}
		} */
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

		if (keys [(unsigned char)'q'])
		{	break;  }

		imshow("depthImageVideo", img);

		// obstacle
		handleObstacles (&depthMap, portID);

		hconcat(rectifiedLeft, rectifiedRight,both);
		imshow("Left and Right",both);

		waitKey(100);
		
	} // end of while

	// Release resources
	capL.release();
	capR.release();
	destroyAllWindows();


	// this never gets reached
	char* strCmd = "STR090\n";
	char* mvCmd = "STP000\n";
	serialPortWrite (strCmd, portID);
	serialPortWrite (mvCmd, portID);
	
	return 0;

} // end of main
