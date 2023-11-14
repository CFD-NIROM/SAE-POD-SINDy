import numpy as np

from scipy import stats
from scipy import signal
from copy import deepcopy as cp
from utils_sindy import *

n_ts = 2000
dt = 0.05
stat = 0.05
endt = 0.05*(n_ts-2)  
taxint = np.arange(stat,endt+1e-5,dt) 





def odeterm(arr,k):
  term = ""
  for i in range(k):
    if arr[i]==0: term = term
    elif arr[i]==1: term = term+"r_"+str(i+1)
    elif arr[i]>1: term = term+"r_"+str(i+1)+"^{"+str(arr[i])+"}"
  return term

# print(odeterm(powers[25],4))
# 'x^2'=odeterm(powers[i])

def show_eqs(xi,n_order,usesin=False,uset=False,usesint=False):
  k = xi.shape[1]
  odeterms = []
  print("k",k)

  usfl = 0
  # p = 0

  powers = Omega(k,n_order)

  if usesin:
    usfl = 2*k
  if usesint:
    usfl = usfl+2
  if uset:
    usfl = usfl+n_order

  term_num = usfl+len(powers)

  if term_num != xi.shape[0]:
    print("term numbers error!")
    return

  for i in range(k):
    p = 0
    eqs_term = xi[:,i]
    eqs = "\\frac{r_"+str(i+1)+"}{dt}="
    for ii in range(len(powers)):
      if eqs_term[ii]==0: eqs=eqs+""
      elif eqs_term[ii]>0: eqs=eqs+"+"+str(eqs_term[ii])+odeterm(powers[ii],k)
      else: eqs=eqs+str(eqs_term[ii])+odeterm(powers[ii],k)
      p = p+1
    if usesin:
      for j in range(k):
        if eqs_term[p]!=0: eqs=eqs+str(eqs_term[p])+"sin(r_"+str(k)+")"
        if eqs_term[p+1]!=0: eqs=eqs+str(eqs_term[p+1])+"cos(r_"+str(k)+")"
        p = p+2
    if uset:
      for jj in range(n_order):
        if eqs_term[p]!=0: eqs=eqs+str(eqs_term[p])+"t^"+str(jj+1)
        p = p+1
    if usesint:
      if eqs_term[p]!=0: eqs=eqs+str(eqs_term[p])+"sin("+str(fre)+"t^"+str(jj+1)+")"
      if eqs_term[p+1]!=0: eqs=eqs+str(eqs_term[p+1])+"cos("+str(fre)+"t^"+str(jj+1)+")"
      p = p+2
    print(eqs)
  return
