This is a grand unifying demo of the python scientific computing environments. It lets you
tinker with kinect data in OpenCV, OpenGL, and Matplotlib, all at the same time!

Installation
------------
1. You need to have installed: IPython, Matplotlib, OpenCV 2.1, PyOpengl, wxPython
2. Build the latest version of libfreenect.
	https://github.com/openkinect/libfreenect
		
3. Build and install the python wrappers for libfreenect

		cd libfreenect/wrappers/python
		python setup.py install

4. Download the latest version of this project

		git clone https://github.com/amiller/libfreenect-goodies.git
		cd pykinect
		
5. Test that python can find libfreenect by running:

		python demo_freenect.py


Usage instructions
------------------

0. Please run this script using:

		ipython -pylab -wthread demo_pykinect.py

1. You should see an opengl window pop up with a preview of a point cloud. You can pan and 
  zoom with the mouse. Run the following commands:

		update()      # Grabs one frame from kinect and updates the point cloud
 		update_on()   # Grabs frames from kinect on a thread - 3d Video! (might be slow!)
		update_off()  # Stops the update thread
  
2. You can also use opencv:

		loopcv()      # Grab frames and display them as a cv image
		(ctrl+c to break)
  
3. You can also use matplotlib:

		imshow(depth)
  
4. Most importantly, you can reload any of the code without pausing or destroying your 
  python instance:

		%run -i demo_pykinect.py
  
  Try changing some of the code, like the downsampling factor (search: downsampling)
  or the point size (search: `GL_POINT_SIZE`) and update the code without quitting python.

This is an ideal environment for developing 3D point cloud algorithms and visualizations.
All of your tools are right at hand. 

**Note to MATLAB users:**  
  Yes, MATLAB already does most of this... there are plenty of reasons to prefer python,
  one of which is access to OpenGL drawing commands - scatter3 isn't adequate!
