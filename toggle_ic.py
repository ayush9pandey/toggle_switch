# Create toggle switch model
from scipy.optimize import fsolve
from scipy.integrate import odeint, quad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ODEs 
def f_ode(x,t, *args):
    K,b_t,b_l,d_t,d_l,del_t,del_l,beta_t,beta_l = args
    m_t = x[0]
    m_l = x[1]
    p_t = x[2]
    p_l = x[3]
    y = [0,0,0,0]
    y[0] = K * b_t/(b_t + p_l) - d_t * m_t
    y[1] = K * b_l/(b_l + p_t) - d_l * m_l
    y[2] = beta_t * m_t - del_t*p_t
    y[3] = beta_l * m_l - del_l * p_l
    return y

x0 = [0,0,0,0]
z0 = [0,0,50,0]
z1 = [0,0,0,50]
z2 = [0,0,100,50]
ic = [x0, z0, z1,z2]
params_nom = [100, 1000, 10, 5, 5, 0.01, 0.01, 0.01, 0.01]
params = tuple(params_nom)

timepoints = np.linspace(0,250,100)
# Solve ODEs, numerically for different ic
for i in range(len(ic)):
        ic0 = ic[i]
        y = odeint(f_ode, ic0, timepoints, args = params)
        plt.subplot(2,len(ic)/2,i+1)
        plt.plot(timepoints, y[:,0],linewidth = 1.5)
        plt.plot(timepoints, y[:,1],linewidth = 1.5)
        plt.plot(timepoints, y[:,2],linewidth = 1.5)
        plt.plot(timepoints, y[:,3],linewidth = 1.5)
        plt.xlabel('Time')
        plt.title('Initial conditions : ' + str(ic0))
        plt.ylabel('States')
plt.show()
