#include <Servo.h>

Servo steeringServo;

int steeringPos = 0;    // variable to store the servo position 
int pinSTR      = 9;
 
String cmdRec = "";
String cmd    = "";
String value  = "";

const int right_enable_a = 3;
const int right_input_a  = 4;
const int right_input_b  = 5;

const int left_enable_a = 6;
const int left_input_a  = 7;
const int left_input_b  = 8;

int portSpeed = 9600;

void ack_turn (int spd, int theta)
{
  // Assuming W/L ratio ~ 1/3
  double wl_const = 1 / ( 3 * tan ( theta * DEG_TO_RAD ) );
  
  double r_v = 0; spd * (1 - 0.5 * (wl_const));
  double l_v = 0; spd * (1 + 0.5 * (wl_const));
  
  if ( (theta * DEG_TO_RAD) < (PI / 2) )
  {
    r_v = spd * (1 + 0.5 * (wl_const));
    l_v = spd * (1 - 0.5 * (wl_const));
    
    Serial.print ("LEFT WHEEL V : ");
  	Serial.println (l_v);
  
  	Serial.print ("RIGHT WHEEL V : ");
  	Serial.println (r_v);
  
  	digitalWrite (left_input_a, HIGH);
  	digitalWrite (left_input_b, LOW);
  	analogWrite (left_enable_a, r_v);
  
  	digitalWrite (right_input_a, HIGH);
  	digitalWrite (right_input_b, LOW);
  	analogWrite (right_enable_a, l_v);
    
  	delay (500);
  }
  else if ( (theta * DEG_TO_RAD) > (PI / 2) )
  {
    r_v = spd * (1 - 0.5 * (wl_const));
    l_v = spd * (1 + 0.5 * (wl_const));
    
    Serial.print ("LEFT WHEEL V : ");
  	Serial.println (l_v);
  
  	Serial.print ("RIGHT WHEEL V : ");
  	Serial.println (r_v);
  
  	digitalWrite (right_input_a, HIGH);
  	digitalWrite (right_input_b, LOW);
  	analogWrite (right_enable_a, r_v);
  
  	digitalWrite (left_input_a, HIGH);
  	digitalWrite (left_input_b, LOW);
  	analogWrite (left_enable_a, l_v);
    
  	delay (500);
  }
}

void straighten ()
{
  steeringPos = 90;
  steeringServo.write (steeringPos);
  
  digitalWrite (left_input_a, LOW);
  digitalWrite (left_input_b, LOW);
  analogWrite (left_enable_a, 0);
  
  digitalWrite (right_input_a, LOW);
  digitalWrite (right_input_b, LOW);
  analogWrite (right_enable_a, 0);
  
  cmd = "";
  value = "";
}



void setup () 
{  
  // seria port setup
  Serial.begin(portSpeed);
  
  pinMode(right_enable_a, OUTPUT);
  pinMode(right_input_a, OUTPUT);
  pinMode(right_input_b, OUTPUT);
  
  pinMode(left_enable_a, OUTPUT);
  pinMode(left_input_a, OUTPUT);
  pinMode(left_input_b, OUTPUT);
    
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
  else if ( cmd == "FWD")
  {
    ack_turn (value.toInt (), 45);
    delay (1000);
  }  
} // --- LOOP END ---

