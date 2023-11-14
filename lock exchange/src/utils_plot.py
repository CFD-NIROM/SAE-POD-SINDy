# import meshio
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from scipy.interpolate import interp1d 
from scipy.interpolate import griddata 
import time

from copy import deepcopy as cp
from sklearn.linear_model import Lasso
from scipy.integrate import solve_ivp
from math import sqrt
from joblib import Parallel, delayed



def contour(Uf,cmp,vml,x,y,fs1=30,cylinder_num=1):

 
  ddx = 0.5*(x[2]-x[1])
  xca = np.linspace(x[0]-ddx, x[len(x)-1]+ddx, len(x)+1)
  
  ddy = 0.5*(y[2]-y[1])
  yca = np.linspace(y[0]-ddy, y[len(y)-1]+ddy, len(y)+1)
  x_min = np.min(x)
  x_max = np.max(x)
  y_min = np.min(y)
  y_max = np.max(y)

  if cmp==None:
    cmp='viridis'  # 'jet'?

  fit = plt.figure(dpi=300,figsize=(4,2))
  ax = plt.axes()

  plt.gca().set_aspect('equal')
  plt.axis([x_min,x_max,y_min,y_max])
  plt.xticks([])
  plt.yticks([])

  if vml==None:
    plt.pcolor(xca,yca,Uf.T,cmap=cmp)
  else:
    plt.pcolor(xca,yca,Uf.T,cmap=cmp,vmin=vml[0],vmax=vml[1])
  plt.xticks(fontsize=fs1-5); plt.yticks(fontsize=fs1-5)
  if cylinder_num==1:

    cx1=np.linspace(8.5,9.5,100)
    cy1=np.sqrt(0.25-(cx1-9)**2)
    cx2=np.linspace(9.5,8.5,100)
    cy2=-np.sqrt(0.25-(cx1-9)**2)
    circx=np.concatenate([cx1,cx2])
    circy=np.concatenate([cy1,cy2])
    plt.plot(circx,circy,color='black',linewidth=1)

    c = patches.Circle(xy=(9, 0), radius=0.5, fc='grey', ec='k')
    ax.add_patch(c)
  elif cylinder_num==3:

    r=0.5

    rx, ry = 6.73, 1.0
    cx11=np.linspace(rx-r,rx+r,100)
    cy11=np.sqrt(r*r-(cx11-rx)**2)+ry
    cx12=np.linspace(rx+r,rx-r,100)
    cy12=-np.sqrt(r*r-(cx12-rx)**2)+ry
    circx=np.concatenate([cx11,cx12])
    circy=np.concatenate([cy11,cy12])
    plt.plot(circx,circy,color='black',linewidth=1)
    c1 = patches.Circle(xy=(rx, ry), radius=r, fc='grey', ec='k')
    ax.add_patch(c1)

    rx, ry = 6.73, -1.0
    cx21=np.linspace(rx-r,rx+r,100)
    cy21=np.sqrt(r*r-(cx21-rx)**2)+ry
    cx22=np.linspace(rx+r,rx-r,100)
    cy22=-np.sqrt(r*r-(cx22-rx)**2)+ry
    circx=np.concatenate([cx21,cx22])
    circy=np.concatenate([cy21,cy22])
    plt.plot(circx,circy,color='black',linewidth=1)
    c2 = patches.Circle(xy=(rx, ry), radius=r, fc='grey', ec='k')
    ax.add_patch(c2)

    rx, ry = 5.0, 0.0
    cx31=np.linspace(rx-r,rx+r,100)
    cy31=np.sqrt(r*r-(cx31-rx)**2)+ry
    cx32=np.linspace(rx+r,rx-r,100)
    cy32=-np.sqrt(r*r-(cx32-rx)**2)+ry
    circx=np.concatenate([cx31,cx32])
    circy=np.concatenate([cy31,cy32])
    plt.plot(circx,circy,color='black',linewidth=1)
    c3 = patches.Circle(xy=(rx, ry), radius=r, fc='grey', ec='k')
    ax.add_patch(c3)
  plt.show()
  return

from scipy.interpolate import griddata


def grid_contour(value,coordinates_orig,limit,cmp,vml,fs1=30,cylinder_num=1,saveif=0,filename=None):


  xmax,xmin,ymax,ymin=limit[0],limit[1],limit[2],limit[3]


  dx, dy = 0.005, 0.005
  Nx = int((xmax-xmin)/dx+1)
  Ny = int((ymax-ymin)/dy+1)
  x = np.linspace(xmin, xmax, Nx)
  y = np.linspace(ymin, ymax, Ny)
  x_mesh, y_mesh = np.meshgrid(x, y)


  grid1 = griddata(coordinates_orig,value,(x_mesh,y_mesh), method='linear', fill_value=0)


  ddx = 0.5*(x[2]-x[1])
  xca = np.linspace(x[0]-ddx, x[len(x)-1]+ddx, len(x)+1)
  ddy = 0.5*(y[2]-y[1])
  yca = np.linspace(y[0]-ddy, y[len(y)-1]+ddy, len(y)+1)
  # vim = [-0.2,1.4]


  fit = plt.figure(dpi=300,figsize=(4,2))
  ax = plt.axes()

  plt.gca().set_aspect('equal')
  plt.axis([xmin,xmax,ymin,ymax])
  plt.xticks([])
  plt.yticks([])

  if cmp==None:
    cmp='viridis'  # 'jet'?

  if vml==None:
    plt.pcolor(xca,yca,grid1[:,:],cmap=cmp)
  else:
    plt.pcolor(xca,yca,grid1[:,:],cmap=cmp,vmin=vml[0],vmax=vml[1])
  plt.xticks(fontsize=fs1-5); plt.yticks(fontsize=fs1-5)  
  if cylinder_num==1:

    cx1=np.linspace(8.5,9.5,100)
    cy1=np.sqrt(0.25-(cx1-9)**2)
    cx2=np.linspace(9.5,8.5,100)
    cy2=-np.sqrt(0.25-(cx1-9)**2)
    circx=np.concatenate([cx1,cx2])
    circy=np.concatenate([cy1,cy2])
    plt.plot(circx,circy,color='black',linewidth=1)

    c = patches.Circle(xy=(9, 0), radius=0.5, fc='grey', ec='k')
    ax.add_patch(c)
  elif cylinder_num==3:

    r=0.5

    rx, ry = 4.23, 0.8
    cx11=np.linspace(rx-r,rx+r,100)
    cy11=np.sqrt(r*r-(cx11-rx)**2)+ry
    cx12=np.linspace(rx+r,rx-r,100)
    cy12=-np.sqrt(r*r-(cx12-rx)**2)+ry
    circx=np.concatenate([cx11,cx12])
    circy=np.concatenate([cy11,cy12])
    plt.plot(circx,circy,color='black',linewidth=1)
    c1 = patches.Circle(xy=(rx, ry), radius=r, fc='grey', ec='k')
    ax.add_patch(c1)
  
    rx, ry = 4.23, -0.8
    cx21=np.linspace(rx-r,rx+r,100)
    cy21=np.sqrt(r*r-(cx21-rx)**2)+ry
    cx22=np.linspace(rx+r,rx-r,100)
    cy22=-np.sqrt(r*r-(cx22-rx)**2)+ry
    circx=np.concatenate([cx21,cx22])
    circy=np.concatenate([cy21,cy22])
    plt.plot(circx,circy,color='black',linewidth=1)
    c2 = patches.Circle(xy=(rx, ry), radius=r, fc='grey', ec='k')
    ax.add_patch(c2)

    rx, ry = 3.0, 0.0
    cx31=np.linspace(rx-r,rx+r,100)
    cy31=np.sqrt(r*r-(cx31-rx)**2)+ry
    cx32=np.linspace(rx+r,rx-r,100)
    cy32=-np.sqrt(r*r-(cx32-rx)**2)+ry
    circx=np.concatenate([cx31,cx32])
    circy=np.concatenate([cy31,cy32])
    plt.plot(circx,circy,color='black',linewidth=1)
    c3 = patches.Circle(xy=(rx, ry), radius=r, fc='grey', ec='k')
    ax.add_patch(c3)
  elif cylinder_num==0:
    plt.xticks(fontsize=fs1-5); plt.yticks(fontsize=fs1-5)
  # plt.colorbar(fraction=0.1,aspect=50,pad=0.05,orientation='horizontal')
  if saveif:
    plt.savefig(filename,bbox_inches='tight')
  plt.show()
  return
