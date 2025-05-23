#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>
#include <errno.h>

#include "serialUtils.h"
#include "cvTest/readData.h"
#include "readParams.h"
#include "makeDepthMapCuda.h"
#include "makeDepthMap.h"
#include "matchingCudaFunt.h"
#include "serialUtils.h"
#include "Structs.h"

#include "utils/imageUtils.h"
#include "utils/matrixUtils.h"
#include "utils/timer.h"
#include "vectorUtils.h"

//motor commands are strings with the following structure. The
//STR186 steer to 186 degrees 0 degrees = hard left 180 degress hard right 90 degrees straight
//RFW128 right forward 128. 255 full speed 0 zero speed
//LBK255 left backwards full speed
//etc

int main ()
{
   const int cmdLength = 7;
   char cmd[cmdLength];
   int portID;
   int bytesWritten;

   while ( strcmp (cmd, "QQQQQQ") != 0)
   {
      portID = serialPortOpen ();
      if ( portID < 0 )
      {
         printf ("Error opening serial port \n");
         exit (0);
      }


      printf ("portID is %d\n", portID);
      printf ("Enter a motor command: ");
      fgets (cmd, sizeof (cmd), stdin);

      printf ("You entered: %s\n", cmd);// Write data to the serial port

      bytesWritten = serialPortWrite (cmd,portID);

      if ( bytesWritten > 0 )
         printf ("Sent %d bytes: %s\n", bytesWritten, cmd);

   }

    /*
      // Set the buffer to zero and read data from the serial port
    memset(buf, 0, sizeof(buf));
    bytesRead = serialPortRead(buf,portID);
    if(bytesRead>0)
	    printf("Received %d bytes: %s\n", bytesRead, buf);
   */

    /* Close the serial port
    if(serialPortClose(portID)<0){
	    printf("Could not close serial port \n");
	    exit(0);
    } else {
	    printf("Serial port closed \n");
    } */

    return 0;

} // END OF MAIN

