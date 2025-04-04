#include <Servo.h>

Servo steeringServo;

int steeringPos = 0;    // variable to store the servo position 
int pinSTR      = 9;
 
String cmdRec = "";
String cmd    = "";
String value  = "";

int portSpeed = 9600;

void straighten ()
{
  steeringPos = 90;
  steeringServo.write (steeringPos);
  cmd = "";
  value = "";
}

void setup () 
{  
  // seria port setup
  Serial.begin(portSpeed);
    
  // steering setup
  pinMode(pinSTR, OUTPUT);
  steeringServo.attach(pinSTR);  // attaches the servo on pin 9 to the servo object  
} // --- SETUP END ---

void loop () 
{
  //read serial port
  if ( Serial.available () ) 
  {      
    cmdRec = Serial.readStringUntil ('\n');
    cmd    = cmdRec.substring (0,3);
    value  = cmdRec.substring (3,6);
    
    //Serial.print("Recieved ");
    Serial.println (cmd);
    //Serial.print("and ");
    Serial.println (value);
  }
  
  // -- Wheel Diameter 2.5625 inches
  // -- Wheel Circumference 8.04625 inches
  // -- 1 degree = 0.02235 inches
  
  
  
  if ( cmd =="STR" )
  {
      steeringPos = value.toInt ();
      steeringServo.write (steeringPos);
      delay (1000);
  }
  else if ( cmd == "STP" )
  {
      straighten ();
      delay (1000);
  }    
} // --- LOOP END ---

