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


def contour(Uf,cmp,vml,x,y,fs1=30,cylinder_num=1,saveif=0,filename=None):
  # ddx = 0.025 dx=0.05
  ddx = 0.5*(x[2]-x[1])
  xca = np.linspace(x[0]-ddx, x[len(x)-1]+ddx, len(x)+1)
  # ddy = 0.025
  ddy = 0.5*(y[2]-y[1])
  yca = np.linspace(y[0]-ddy, y[len(y)-1]+ddy, len(y)+1)
  x_min = np.min(x)
  x_max = np.max(x)
  y_min = np.min(y)
  y_max = np.max(y)
  # cmp setting
  if cmp==None:
    cmp='viridis'  # 'jet'

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
  plt.xticks(fontsize=fs1-5); plt.yticks(fontsize=fs1-5)  # 【？】
  if cylinder_num==1:
    
    # cx1=np.linspace(8.5,9.5,100)
    # cy1=np.sqrt(0.25-(cx1-9)**2)
    # cx2=np.linspace(9.5,8.5,100)
    # cy2=-np.sqrt(0.25-(cx1-9)**2)
    # circx=np.concatenate([cx1,cx2])
    # circy=np.concatenate([cy1,cy2])
    # plt.plot(circx,circy,color='black',linewidth=1)

    c = patches.Circle(xy=(9, 0), radius=0.5, fc='white')
    ax.add_patch(c)
  # plt.colorbar(fraction=0.04,aspect=50,pad=0.07,orientation='horizontal')
  # plt.colorbar(fraction=0.04,aspect=50,pad=0.07)
  # if saveif:
    # plt.savefig(filename,bbox_inches='tight')
  plt.axis('off')
  plt.show()
  return