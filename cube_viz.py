import argparse
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

parser = argparse.ArgumentParser()
parser.add_argument('cubefiles',nargs='+')
parser.add_argument('--tmax',nargs='?',type=float,default=10.)

args = parser.parse_args()

app = QtGui.QApplication([])
w = gl.GLViewWidget()

# g = gl.GLGridItem()
# g.scale(10, 10, 1)
# w.addItem(g)

colours = ((255,0,255),(0,255,255),(255,255,0))
colour_idx = -1;
t_max = args.tmax;

for filename in args.cubefiles:
    colour_idx = (colour_idx + 1) % len(colours)
    print('plotting: ',filename,colours[colour_idx])
    cube = np.load(filename)
    x,y,z,t = [cube[i] for i in 'xyzt']
    w.opts['distance'] = (len(x)**2+len(y)**2+len(z)**2)**0.5
    t = np.rollaxis(t,-1)
    t_norm = np.clip(t,0,t_max)/t_max
    t_gl = np.empty(t.shape+(4,),dtype=np.ubyte)
    t_gl[...,0] = t_norm*colours[colour_idx][0]
    t_gl[...,1] = t_norm*colours[colour_idx][1]
    t_gl[...,2] = t_norm*colours[colour_idx][2]
    t_gl[...,3] = np.max(t_gl[...,0:3],axis=-1)/4.2
    t_gl[:,0,0] = [255,0,0,255]
    t_gl[0,:,0] = [0,255,0,255]
    t_gl[0,0,:] = [0,0,255,255]
    t_gl[:,0,-1] = [255,0,0,255]
    t_gl[0,:,-1] = [0,255,0,255]
    t_gl[0,-1,:] = [0,0,255,255]
    t_gl[:,-1,0] = [255,0,0,255]
    t_gl[-1,:,0] = [0,255,0,255]
    t_gl[-1,0,:] = [0,0,255,255]
    t_gl[:,-1,-1] = [255,0,0,255]
    t_gl[-1,:,-1] = [0,255,0,255]
    t_gl[-1,-1,:] = [0,0,255,255]

    v = gl.GLVolumeItem(t_gl,smooth=True,glOptions='additive')
    v.translate(-len(z)//2,-len(x)//2,-len(y)//2)
    w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)

w.show()
app.exec_()
