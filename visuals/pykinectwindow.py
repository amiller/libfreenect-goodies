# This module requires IPython to work! It is meant to be used from an IPython environment with: 
#   ipython -wthread and -pylab
# See demo_pykinect.py for an example

import wx
from wx import glcanvas
from OpenGL.GL import *

# Get the app ourselves so we can attach it to each window
if not '__myapp' in wx.__dict__:
  wx.__myapp = wx.PySimpleApp()
app = wx.__myapp

class Window(wx.Frame):
  
    # wx events can be put in directly
    def eventx(self, target):
        def wrapper(*args, **kwargs):
          target(*args, **kwargs)
        self.canvas.Bind(wx.__dict__[target.__name__], wrapper)
  
    # Events special to this class, just add them this way
    def event(self, target):
        def wrapper(*args, **kwargs):
          target(*args, **kwargs)   
        self.__dict__[target.__name__] = wrapper
        
    def _wrap(self, name, *args, **kwargs):
      try:
        self.__getattribute__(name)
      except AttributeError:
        pass
      else:
        self.__getattribute__(name)(*args, **kwargs)
            

    def __init__(self, title='WxWindow', id=-1, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE,
                 name='frame'):
                 
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE

        super(Window,self).__init__(None, id, title, pos, size, style, name)

        attribList = (glcanvas.WX_GL_RGBA, # RGBA
                      glcanvas.WX_GL_DOUBLEBUFFER, # Double Buffered
                      glcanvas.WX_GL_DEPTH_SIZE, 24) # 24 bit
              
        self.canvas = glcanvas.GLCanvas(self, attribList=attribList)

        self.canvas.Bind(wx.EVT_ERASE_BACKGROUND, self.processEraseBackgroundEvent)
        self.canvas.Bind(wx.EVT_SIZE, self.processSizeEvent)
        self.canvas.Bind(wx.EVT_PAINT, self.processPaintEvent)
        self.Show()

    def processEraseBackgroundEvent(self, event):
        """Process the erase background event."""
        pass # Do nothing, to avoid flashing on MSWin

    def processSizeEvent(self, event):
        """Process the resize event."""
        if self.canvas.GetContext():
            # Make sure the frame is shown before calling SetCurrent.
            #self.Show()
            self.canvas.SetCurrent()
            size = self.GetClientSize()
            self.OnReshape(size.width, size.height)
            #self.canvas.Refresh(False)
        event.Skip()

    def processPaintEvent(self, event=None):
        """Process the drawing event."""
        self.canvas.SetCurrent()
        self._wrap('on_draw')
        self.canvas.SwapBuffers()
        if event: event.Skip()

    def OnReshape(self, width, height):
        """Reshape the OpenGL viewport based on the dimensions of the window."""
        glViewport(0, 0, width, height)