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


usesint = True
dtsnap = 0.05
n_ts = 2000
taxx=np.arange(0,dtsnap*n_ts,dtsnap) 
tax=np.arange(dtsnap,dtsnap*(n_ts-1)-1e-5,dtsnap)  # 1000

# tax = np.arange(dtsnap,dtsnap*(n_ts-1)-1e-5,dtsnap)
fre = 0.23034551827741614
delta = 1 + 2  
n_order = 3
usesin = False
uset = True



# test_data = np.array([[i+j for i in range(1,3)]for j in range(1,4)])
import operator
import itertools

def Omega(k, n):
  if k == 1:
    return [(i,) for i in range(n + 1)]
  else:
    powers = []
    for i in range(n + 1):
      for j in Omega(k - 1, n - i):
        powers.append((i,) + j)
    #return powers
    return sorted(powers, reverse=False)

def RHS_rk(data,n_order,usesin=False):  # no derivate term
  n, k = data.shape
  usfl = 0
  p = 0

  if usesin:
    usfl = 2*k
  if usesint:
    usfl = usfl+2
  if uset:
    usfl = usfl+n_order

  rhs_functions = {}  # dict
  powers = Omega(k,n_order)
  f = lambda x,y: np.prod(np.power(list(x),list(y)))  # (x1^y1)*(x2^y2)*...
  for power in powers:
    rhs_functions[power] = [lambda x,y=power: f(x,y), power]
    # dict rhs_fun的key为不可变元素tuple(power),指向[f,power]

  # print(p,len(powers))
  # RHS = np.ones((n,1+len(powers)+usfl),dtype=np.float64)
  RHS = np.ones((n,len(powers)+usfl),dtype=np.float64)


  l = 0

  for pw in rhs_functions.keys():
    func, power = rhs_functions[pw][0], rhs_functions[pw][1]
    # print(power)
    new_column = np.zeros((n,))
    for i in range(n):
      new_column[i] = func(data[i,:],power)  
    RHS[:,l]=new_column
    l=l+1

  if usesin:
    for i in range(k):
      RHS[:,l] = np.sin(data[:,i])
      RHS[:,l+1] = np.cos(data[:,i])
      l = l+2
  if uset:
    for iii in range(n_order):
      RHS[:,l+iii] = np.power(tax,iii+1)
    l = l+n_order

  if usesint:
    s = 2*np.pi*fre
    RHS[:,l] = np.sin(s*tax)
    RHS[:,l+1] = np.cos(s*tax)
    l = l+2


  # print(RHS.shape)
  # print(powers)

  return RHS


l_max_iter=1000 # Default
l_tol=0.0001 # Default

def alasso(RHS, LHS, alr):
  # params
  n_lasso_iterations = 100  
  tol = 1e-10
  absl = lambda w: (np.abs(w) ** delta + 1e-30) 

  n_samples, n_features = RHS.shape  
  weights = np.ones(n_features)
  for k in range(n_lasso_iterations):
    prevw = cp(weights)
    RHS_w = RHS / weights[np.newaxis, :]  
    clf = Lasso(alpha=alr, fit_intercept=False, max_iter=l_max_iter, tol=l_tol)  
    clf.fit(RHS_w, LHS)  
    coef_ = clf.coef_ / weights  
    weights = 1 / absl(coef_)  
    if np.mean((weights - prevw) ** 2) < tol:  
  return coef_


def tlsa_norm(RHS, LHS, alpha, lamda=10 ** -2, iter=100): # , normalize=2):
  n, d = RHS.shape
  if lamda != 0:
    w = np.linalg.lstsq(RHS.T.dot(RHS) + lamda * np.eye(d), RHS.T.dot(LHS), rcond=None)[0]
  else:
    w = np.linalg.lstsq(RHS, LHS, rcond=None)[0]

  bigcoeffs = np.where(abs(w) > alpha)[0]

  relevant_coeff_num = d

  for it in range(iter):
    smallcoeffs = np.where(abs(w) <= alpha)[0]
    new_bigcoeffs = [i for i in range(d) if i not in smallcoeffs]
    if relevant_coeff_num == len(new_bigcoeffs):
      break  
    else:
      relevant_coeff_num = len(new_bigcoeffs)
    if len(new_bigcoeffs) == 0:
      if it == 0:
        print('Tolenrance too high, all coefficients set below tolerance')
        return w
      else:
        break
    bigcoeffs = new_bigcoeffs 
    w[smallcoeffs] = 0
    if lamda != 0:
      w[bigcoeffs] = np.linalg.lstsq(RHS[:, bigcoeffs].T.dot(RHS[:, bigcoeffs]) + lamda * np.eye(relevant_coeff_num),
                                     RHS[:, bigcoeffs].T.dot(LHS), rcond=None)[0]
    else:
      w[bigcoeffs] = np.linalg.lstsq(RHS[:, bigcoeffs], LHS)[0]
  if len(bigcoeffs) != 0:  
    w[bigcoeffs] = np.linalg.lstsq(RHS[:, bigcoeffs], LHS, rcond=None)[0]
    return w
  else:
    return w




