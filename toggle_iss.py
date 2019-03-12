# Create toggle switch model
from scipy.optimize import fsolve
from scipy.integrate import odeint, quad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ODEs 
def f_ode(x,t, *args):
    K,b_t,b_l,d_t,d_l,del_t,del_l,beta_t,beta_l, Kt0, Kl0 = args
    m_t = x[0]
    m_l = x[1]
    p_t = x[2]
    p_l = x[3]
    md_t = x[4]
    md_l = x[5]
    pd_t = x[6]
    pd_l = x[7]
    y = np.zeros(8)
    y[0] = K * b_t**2/(b_t**2 + p_l**2) - d_t * m_t 
    y[1] = K * b_l**2/(b_l**2 + p_t**2) - d_l * m_l
    y[2] = beta_t * m_t - del_t*p_t
    y[3] = beta_l * m_l - del_l * p_l

    y[4] = K * b_t**2/(b_t**2 + pd_l**2) - d_t * md_t + Kt0
    y[5] = K * b_l**2/(b_l**2 + pd_t**2) - d_l * md_l + Kl0
    y[6] = beta_t * md_t - del_t*pd_t
    y[7] = beta_l * md_l - del_l * pd_l
    return y

x0 = np.zeros(8)
params_nom = [100, 1000, 10, 5, 5, 0.01, 0.01, 0.01, 0.01] # default params
params_nom.append(10) # leak params
params_nom.append(20) # leak params

timepoints = np.linspace(0,450,100)
params = tuple(params_nom)
# Solve ODEs, numerically
y = odeint(f_ode, x0, timepoints, args = params)

# Plot solutions from numerical simulation
plt.subplot(2,2,1)
plt.plot(timepoints,y[:,0], label = 'Original model')
plt.plot(timepoints,y[:,4], label = 'Perturbed model')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Species amount')
plt.title('mRNA TetR response')

plt.subplot(2,2,2)
plt.plot(timepoints,y[:,1], label = 'Original model')
plt.plot(timepoints,y[:,5], label = 'Perturbed model')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Species amount')
plt.title('mRNA LacI response')

plt.subplot(2,2,3)
plt.plot(timepoints,y[:,2], label = 'Original model')
plt.plot(timepoints,y[:,6], label = 'Perturbed model')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Species amount')
plt.title('Protein TetR response')

plt.subplot(2,2,4)
plt.plot(timepoints,y[:,3], label = 'Original model')
plt.plot(timepoints,y[:,7], label = 'Perturbed model')
plt.title('Protein LacI response')
plt.xlabel('Time')
plt.ylabel('Species amount')
plt.legend()


plt.show()
