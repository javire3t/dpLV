# Lotka-Voltera dynamic programming v 2.0 by Javier Solano
import numpy as np
import  numpy.matlib 
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Cost function
def cost(vx,vu,deltat):
  return -np.minimum(vx,vu*deltat);

# State model
def model(vx,vu,deltat):
  return np.maximum(0,vx+deltat*(0.02*(vx-vx*vx/1000)-vu));

# State Variable
param_x_t=1001;param_x_max = 1000;param_x_min = 0;
v_x=np.linspace(param_x_min,param_x_max,param_x_t);

# Control 
param_u_t=101;param_u_max = 10;param_u_min = 0;
v_u=np.linspace(param_u_min,param_u_max,param_u_t);

# Tiempo
param_t_max = 200+1e-6;#Not as in Matlab !
param_t_min = 0;param_t_delta=0.2;
v_t=np.arange(param_t_min,param_t_max,param_t_delta);param_t_t=len(v_t)

# Final state constraints and penalization
param_x_final=750; # Constraint
param_coutMAX=750; # Penalisation - this is very important

#Initialice Uopt and Jmin matrix
Uopt=np.zeros((param_x_t,param_t_t))
Jmin=np.zeros((param_x_t,param_t_t))


# DP algorithm initialization
VUX, VXU = np.meshgrid(v_u, v_x, indexing='ij'); # Vectorisation
alg_cost2goK1=np.zeros((param_x_t), dtype=int); # Cost initialization
alg_cost2goK1[v_x<param_x_final]=param_coutMAX;# final state penalisation

# DP algorithm iteration
for x in range(param_t_t-1, -1, -1): # Range es distinto que en MAtlab
  xk1=model(VXU,VUX,param_t_delta)
  Jminxk1=np.interp(xk1,v_x,alg_cost2goK1);
  h=cost(VXU,VUX,param_t_delta);
  alg_cost2goK1=np.amin(Jminxk1+h, axis=0)
  position=np.argmin(Jminxk1+h, axis=0)
  Uopt[:,x]=v_u[position];
  Jmin[:,x]=alg_cost2goK1;

# DP algorithm initialization

#Simulation Forward
fw_t=v_t;fw_x=np.zeros(len(fw_t));fw_P=np.zeros(len(fw_t));fw_u=np.zeros(len(fw_t));
fw_x[0]=250; # Initial state value
f = interp2d(v_t,v_x,Uopt, kind='linear')# distinto que en Matlab. Definir la interpolación
for k in range(0,param_t_t-1):  
    fw_u[k]=f(fw_t[k],fw_x[k]);
    fw_x[k+1]=model(fw_x[k],fw_u[k],param_t_delta);
    fw_P[k+1]=fw_P[k]+fw_u[k]*param_t_delta;


plt.plot(fw_t,fw_u)
plt.grid()
plt.title("Fishing rate")
plt.ylabel('fish/day')
plt.xlabel('day')
plt.show()

plt.plot(fw_t,fw_P)
plt.grid()
plt.title("Captured fish")
plt.ylabel('fish')
plt.xlabel('day')
plt.show()

plt.plot(fw_t,fw_x)
plt.grid()
plt.title("Fish in the pond")
plt.ylabel('fish')
plt.xlabel('day')
plt.show()


ax = plt.subplot()
im = plt.pcolormesh(v_t,v_x,Uopt)
plt.title("Optimal control")
plt.ylabel('fish')
plt.xlabel('day')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()


ax = plt.subplot()
im =plt.pcolormesh(v_t, v_x, -Jmin)
plt.title("Maximal captured fish")
plt.ylabel('fish')
plt.xlabel('day')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()