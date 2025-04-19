#include <Servo.h>

Servo steeringServo;

const int right_enable_a = 3;
const int right_input_a  = 4;
const int right_input_b  = 5;

const int left_enable_a = 6;
const int left_input_a  = 7;
const int left_input_b  = 8;
const int pinSTR        = 9;
 
String cmdRec = "";
String cmd    = "";
String value  = "";

int portSpeed = 9600;
int steeringPos = 0;    // variable to store the servo position

void ack_turn (int spd, int theta)
{
   double wl_const = (16.25) / ( (23.50) * tan ( theta * DEG_TO_RAD ) );
  
   double r_v = 0; spd * (1 - 0.5 * (wl_const));
   double l_v = 0; spd * (1 + 0.5 * (wl_const));
  
   // if ( (theta * DEG_TO_RAD) < (PI / 2) )
   if ( theta < 90 )
   {
      r_v = spd * (1 + 0.5 * (wl_const));
      l_v = spd * (1 - 0.5 * (wl_const));
    
      Serial.print ("LEFT WHEEL VELOCITY : ");
      Serial.println (l_v);
  
      Serial.print ("RIGHT WHEEL VELOCITY : ");
      Serial.println (r_v);
      
      steeringPos = value.toInt ();
      steeringServo.write (theta);
  
      digitalWrite (left_input_a, HIGH);
      digitalWrite (left_input_b, LOW);
      analogWrite (left_enable_a, r_v);
  
      digitalWrite (right_input_a, HIGH);
      digitalWrite (right_input_b, LOW);
      analogWrite (right_enable_a, l_v);
    
      delay (500);
   }
   // else if ( (theta * DEG_TO_RAD) > (PI / 2) )
   else if ( theta > 90 )
   {
      r_v = spd * (1 - 0.5 * (wl_const));
      l_v = spd * (1 + 0.5 * (wl_const));
    
      Serial.print ("LEFT WHEEL VELOCITY : ");
      Serial.println (l_v);
  
      Serial.print ("RIGHT WHEEL VELOCITY : ");
      Serial.println (r_v);
      
      steeringPos = value.toInt ();
      steeringServo.write (theta);
  
      digitalWrite (right_input_a, HIGH);
      digitalWrite (right_input_b, LOW);
      analogWrite (right_enable_a, r_v);
  
      digitalWrite (left_input_a, HIGH);
      digitalWrite (left_input_b, LOW);
      analogWrite (left_enable_a, l_v);
    
      delay (500);
   }
   else 
   {
      // -- Wheel Diameter 2.5625 inches
      // -- Wheel Circumference 8.04625 inches | 0.2043 m
      // -- 1 degree = 0.02235 inches

      r_v = spd * (1 - 0.5 * (wl_const));
      
      Serial.print ("CURRENT VELOCITY : ");
      Serial.println (r_v);

      digitalWrite (right_input_a, HIGH);
      digitalWrite (right_input_b, LOW);
      analogWrite (right_enable_a, r_v);
   
      digitalWrite (left_input_a, HIGH);
      digitalWrite (left_input_b, LOW);
      analogWrite (left_enable_a, r_v);

      delay (500);
   }
   
   cmd = "";
   value = "";
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


void reverse (int spd)
{
   digitalWrite (left_input_a, LOW);
   digitalWrite (left_input_b, HIGH);
   analogWrite (left_enable_a, spd);
   
   digitalWrite (right_input_a, LOW);
   digitalWrite (right_input_b, HIGH);
   analogWrite (right_enable_a, spd);
   
   cmd = "";
   value = "";
} 


void setup () 
{  
   // serial port setup
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
    
   if ( cmd == "STR" )
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
      ack_turn (value.toInt (), 90);
      delay (1000);
   }
   else if ( cmd == "REV" )
   {
      reverse (value.toInt ());
      delay (1000);
   }
   else if ( cmd == "SSS" )
   {
       int angle = 0;
       
       for (angle = 21; angle <= 160; angle += 10)
       {
         ack_turn (value.toInt (), angle);
         delay (750);
       }
       
       for (angle = 160; angle >= 21; angle -= 10)
       {
         ack_turn (value.toInt (), angle);
         delay (750);
       }
       
       delay (1000);
   }
   else
   {
      straighten ();
      delay (1234);
   }  
} // --- LOOP END ---

