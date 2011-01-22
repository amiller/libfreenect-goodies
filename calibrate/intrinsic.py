import freenect 
import numpy as np
import cv

"""
    If you need to do intrinsic calibration, I've left that up to the matlab calibration toolbox.
    You can use record() to save IR images. 
""" 

# This is the directory to store ir frames
dirname = 'calibdata/calib_ir_2'

def grab():
  global depth, ir
  ir,_ = freenect.sync_get_video(0, freenect.VIDEO_IR_8BIT)

def record():
  for i in range(0,100):
    while 1:
      grab()
      showimagegray('ir', ir)  
      if cv.WaitKey(10) > -1:      
        filename = '%s/ir_%05d.jpg' % (dirname, i)
        cv.SaveImage(filename, ir) 
        print 'saved ', i
        break
