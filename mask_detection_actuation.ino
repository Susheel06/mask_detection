#include <Servo.h>
Servo myservo; // create servo object to control a servo

int x;
int pos = 0;

void setup() {
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
  Serial.begin(9600);
  Serial.setTimeout(1);
}

void loop() {
  while (!Serial.available());
  x = Serial.readString().toInt();
  Serial.print(x);

  if(x == 0){
    for (pos = 0; pos <= 90; pos += 1) { // goes from 0 degrees to 90 degrees
      // in steps of 1 degree
      myservo.write(pos);              // tell servo to go to position in variable 'pos'
      delay(800);                       // waits 0.8s for the servo to reach the position
    }
    
    delay(3000);
    
    for (pos = 90; pos >= 0; pos -= 1) { // goes from 90 degrees to 0 degrees
      myservo.write(pos);              // tell servo to go to position in variable 'pos'
      delay(800);                       // waits 0.8s for the servo to reach the position
    }
  } 
}
