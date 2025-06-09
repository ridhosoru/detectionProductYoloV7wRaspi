from time import sleep
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(24, GPIO.OUT)
GPIO.output(24,GPIO.HIGH)


GPIO.setup(25,GPIO.OUT)
GPIO.output(25,GPIO.HIGH)


sensorp=16
GPIO.setup(sensorp,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
try:
    while True:
        if GPIO.input(sensorp):
            GPIO.output(24,0)
            print("1")
            #GPIO.output(25,0)

        else:
            print("0")
            GPIO.output(24,1)
            #  GPIO.output(25,1)
except KeyboardInterrupt:
    print("end")

     