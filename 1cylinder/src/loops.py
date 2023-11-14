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

from scipy import signal

from utils_sindy import *

import timeout_decorator


n_ts = 2000
dt = 0.05
stat = 0.05
endt = 0.05*(n_ts-2)  

taxint = np.arange(stat,endt+1e-5,dt) 
dtff = cp(dt)



@timeout_decorator.timeout(100)
def cal_sindy_k(n,data,ddata,alr,regression='alasso'):
  alpha = alr[n]
  k = data.shape[1]
  usesin = False

  LHS = cp(ddata)
  RHS = RHS_rk(data,n_order,usesin)

  xi1 = np.zeros((len(RHS[0,:]),len(LHS[0,:])))
  if regression=='alasso':
    for i in range(xi1.shape[1]):
      xi1[:, i] = alasso(RHS, LHS[:,i], alr[n])  
  elif regression=='tlsa_norm':
    for i in range(xi1.shape[1]):
      xi1[:, i] = tlsa_norm(RHS, LHS[:,i], alr[n])
  else:
    print('regression no exist!')
    return
  xi = cp(xi1)  
  xi = xi.astype(np.float16)  
  def sindy_fun(t,y):
    re_r = np.zeros((len(LHS[0,:],)))
    for j in range(len(LHS[0,:])): 
      re_r[j] = y[j]
    ii = 0
    usfl = 0
    if usesin:
      usfl = 2*k
    if usesint:
      usfl = usfl + 2
    if uset:
      usfl = usfl + n_order

    rhs_functions = {}  # dict
    powers = Omega(k,n_order)
    f = lambda x, y: np.prod(np.power(list(x), list(y)))  
    for power in powers:
      rhs_functions[power] = [lambda x, y=power: f(x, y), power]
     
    re_rhs = np.ones([len(powers) + usfl]) 

    ii = 0
    for pw in rhs_functions.keys():
      func, power = rhs_functions[pw][0], rhs_functions[pw][1]
      re_rhs[ii] = func(re_r,power)
      ii = ii+1

    if usesin:
      for i in range(k):
        re_rhs[ii] = np.sin(re_r[i])
        re_rhs[ii+1] = np.cos(re_r[i])
        ii = ii+2
    if uset:
      for iii in range(n_order):
        re_rhs[ii+iii] = np.power(t,iii+1)
      ii = ii+n_order
    if usesint:  
      s = 2*np.pi*fre
      re_rhs[ii] = np.sin(s*t)
      re_rhs[ii+1] = np.cos(s*t)
      ii = ii+2
    reconstx = np.dot(re_rhs,xi)  

    return (list(reconstx))
  data0 = data[0,:]
  v0 = list(data0)
  t = taxint
  vodeivp = solve_ivp(sindy_fun,[stat, endt+1e-05], v0, method='RK45', t_eval=taxint)
  vodet = vodeivp.t   
  vodey = vodeivp.y  

  draw = 1
  if len(vodet)<len(data):
    draw = 2
    print('Unexpect Error: solve ivp wrong!')
  print("count_nonzero(xi): ",np.count_nonzero(xi))

  return alpha, xi, vodet, vodey, draw



def loop5(n,data,ddata,alr,regression='alasso'):
  try:
    alpha, xi, vodet, vodey, draw = cal_sindy_k(n,data,ddata,alr,regression)
    if draw==2:
      sig = 0  
    num_r1=np.count_nonzero(xi[:,0])
    num_r2=np.count_nonzero(xi[:,1])
    num_r3=np.count_nonzero(xi[:,2])
    num_r4=np.count_nonzero(xi[:,3])
    num_r5=np.count_nonzero(xi[:,4])

  except:
    alpha=alr[n]
    xi, vodet, vodey, num_r1, num_r2, num_r3, num_r4, num_r5, draw=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    print('Time out')
  return alpha, xi, vodet, vodey, num_r1, num_r2, num_r3, num_r4, num_r5, draw

def loop4(n,data,ddata,alr,regression='alasso'):
  try:
    alpha, xi, vodet, vodey, draw = cal_sindy_k(n,data,ddata,alr,regression)
    if draw==2:
      sig = 0  
    num_r1=np.count_nonzero(xi[:,0])
    num_r2=np.count_nonzero(xi[:,1])
    num_r3=np.count_nonzero(xi[:,2])
    num_r4=np.count_nonzero(xi[:,3])
    aic = AIC(data,vodey,vodet,xi)
    mean1,rms1 = rms_and_mean(vodey,vodet)
    # nycmp[n]=np.count_nonzero(xi[:,1])
    # nx=nxcmp[n]
    # ny=nycmp[n]
  except:
    alpha=alr[n]
    xi, vodet, vodey, num_r1, num_r2, num_r3, num_r4, draw=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    aic = np.nan
    mean1,rms1 = np.nan,np.nan
    print('Time out')
  return alpha, xi, vodet, vodey, num_r1, num_r2, num_r3, num_r4, draw, aic, mean1, rms1

def loop3(n,data,ddata,alr,regression='alasso'):
  try:
    alpha, xi, vodet, vodey, draw = cal_sindy_k(n,data,ddata,alr,regression)
    if draw==2:
      sig = 0  
    num_r1=np.count_nonzero(xi[:,0])
    num_r2=np.count_nonzero(xi[:,1])
    num_r3=np.count_nonzero(xi[:,2])
    aic = AIC(data,vodey,vodet,xi)
    mean1,rms1 = rms_and_mean(vodey,vodet)
    # num_r4=np.count_nonzero(xi[:,3])
    # nycmp[n]=np.count_nonzero(xi[:,1])
    # nx=nxcmp[n]
    # ny=nycmp[n]
    L2_error1 = np.linalg.norm(vodey.T-data)/np.linalg.norm(data)
    fre1 = strouhal(vodey,dt)
  except:
    alpha=alr[n]
    xi, vodet, vodey, num_r1, num_r2, num_r3, draw=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    aic = np.nan
    mean1,rms1 = np.nan,np.nan
    L2_error1 = np.nan
    fre1 = np.nan
    print('Time out')
  return alpha, xi, vodet, vodey, num_r1, num_r2, num_r3, draw, aic, mean1, rms1, L2_error1, fre1

def loop2(n,data,ddata,alr,regression='alasso'):
  try:
    alpha, xi, vodet, vodey, draw = cal_sindy_k(n,data,ddata,alr,regression)
    if draw==2:
      sig = 0  
    num_r1=np.count_nonzero(xi[:,0])
    num_r2=np.count_nonzero(xi[:,1])
    # num_r3=np.count_nonzero(xi[:,2])
    # num_r4=np.count_nonzero(xi[:,3])
    # nycmp[n]=np.count_nonzero(xi[:,1])
    # nx=nxcmp[n]
    # ny=nycmp[n]
    aic = AIC(data,vodey,vodet,xi)
    mean1,rms1 = rms_and_mean(vodey,vodet)
    L2_error1 = np.linalg.norm(vodey.T-data)/np.linalg.norm(data)
    fre1 = strouhal(vodey,dt)
  except:
    alpha=alr[n]
    xi, vodet, vodey, num_r1, num_r2, draw=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    aic = np.nan
    mean1,rms1 = np.nan,np.nan
    L2_error1 = np.nan
    fre1 = np.nan
    print('Time out')
  return alpha, xi, vodet, vodey, num_r1, num_r2, draw, aic, mean1, rms1, L2_error1, fre1

def AIC(data,vodey,vodet,xi):

  newy = np.zeros_like(data)
  ii = 0
  for i2 in range(len(vodet)):
    if (ii < len(taxint) and np.abs(taxint[ii] - vodet[i2]) < 0.001):
      newy[ii, :] = vodey[:, i2]
      ii = ii + 1
#     print(ii)
  if (ii == len(taxint)):
    rss = np.sum((data - newy) ** 2)
    aic1 = len(taxint) * np.log(rss / len(taxint))  # likelihood
    aic2 = (np.count_nonzero(xi)) * 2  # num of coefs.  print("count_nonzero(xi): ",np.count_nonzero(xi))
    aic = aic1 + aic2
  else:
    aic = 10 ** 10  # 10^10
  return aic


def rms_and_mean(vodey,vodet): 
  data = vodey.T
  tempval=len(data)
  k = data.shape[1]

  meanx=np.mean(data,axis=0)
  mean1=np.sum(meanx)

  msx=np.zeros(k)
  rmsx=np.zeros(k)
  for i in range(k):
    msx[i]=np.sum((data[-tempval:,i]-meanx[i])**2)
    rmsx[i]=np.sqrt(msx[i]/tempval)
  rms1=np.sum(rmsx)

  # print(rms1,mean1)
  # print(msx)
  # print(meanx)

  return mean1,rms1


def strouhal(vodey,dtff): 
  fft_target = signal.hann(len(vodey[0, :])) * vodey[0, :]
  F = np.fft.fft(fft_target)  
  Amp = np.abs(F) 
  N1 = len(vodey[1, :])  
  freq = np.linspace(0, 1.0 / dtff, N1)  
  return starray