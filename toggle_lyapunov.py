# Create toggle switch model
from scipy.optimize import fsolve
from scipy.integrate import odeint, quad
from scipy.linalg import eigvals, solve_lyapunov
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
    y[0] = K * b_t**2/(b_t**2 + p_l**2) - d_t * m_t
    y[1] = K * b_l**2/(b_l**2 + p_t**2) - d_l * m_l
    y[2] = beta_t * m_t - del_t*p_t
    y[3] = beta_l * m_l - del_l * p_l
    return y

def ode_fun(x,*args):
    K,b_t,b_l,d_t,d_l,del_t,del_l,beta_t,beta_l = args
    m_t = x[0]
    m_l = x[1]
    p_t = x[2]
    p_l = x[3]
    y = [0,0,0,0]
    y[0] = K * b_t**2/(b_t**2 + p_l**2) - d_t * m_t
    y[1] = K * b_l**2/(b_l**2 + p_t**2) - d_l * m_l
    y[2] = beta_t * m_t - del_t*p_t
    y[3] = beta_l * m_l - del_l * p_l
    return y


x0 = [0,0,0,0]
params_nom = [100, 10, 100, 5, 5, 0.01, 0.01, 0.01, 0.01]
# params_nom = [0.01, 0.01, 0.01, 1, 1, 0.1, 0.1, 1, 1]

timepoints = np.linspace(0,10,100)
params = tuple(params_nom)
# Solve ODEs, numerically
y = odeint(f_ode, x0, timepoints, args = params)

# Calculate Lipscitz constant
K,b_t,b_l,d_t,d_l,del_t,del_l,beta_t,beta_l = params_nom
eqpts = fsolve(ode_fun, x0, args = params)
m_te, m_le, p_te, p_le = eqpts
print('The equilibrium point for the system is {0}'.format(eqpts))
# Jacobian
J = np.array([[-d_t, 0, 0, -K*b_t/((b_t + p_le)**2)], [0, -d_l, -K*b_l/( (b_l + p_te)**2),0], [beta_t, 0, -del_t, 0], [0, beta_l, 0, -del_l] ])
print('The eigen values of the linearized system are {0}'.format(eigvals(J)))
Q = -1 * np.identity(4)
P = solve_lyapunov(J,Q)
print('The Lyapunov matrix is {0}'.format(P))
print('The eigen values of the Lyapunov matrix are {0}'.format(eigvals(P)))

# Symbolically calculate V(x) = x^T P x
import sympy as sym
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')
x3 = sym.Symbol('x3')
x4 = sym.Symbol('x4')
x = sym.Matrix([x1,x2,x3,x4])
Vx = x.transpose() * (P * x)
print('The Lyapunov function is {0}'.format(Vx))

eigQ_min = np.abs(min(eigvals(-1 *Q)))
eigP_max = np.abs(max(eigvals(P)))
N = eigQ_min/(2*eigP_max)
x3sv = np.linspace(0.01,0.09,100)
x4sv = np.linspace(0.02,0.09,100)
max_store = 0
for x3s, x4s in zip(x3sv, x4sv):
    a1 = - ( K * b_t**2 / x4s) * ( 1/(b_t**2 + p_le**2) - 1/(b_t**2 + (x4s + p_le)**2) ) + ( 2 *K * p_le * b_t**2) / ( (b_t**2 + p_le**2)**2)
    a1 = np.abs(a1)
    b1 = - ( K * b_l**2 / x3s) * ( 1/(b_l**2 + p_te**2) - 1/(b_l**2 + (x3s + p_te)**2) ) + ( 2 *K * p_te * b_l**2) / ( (b_l**2 + p_te**2)**2)
    b1 = np.abs(b1)
    diff_norm = max(a1,b1)
    if diff_norm < N:
        if a1 > b1:
            max_store = max(x4s, max_store)
        if b1 > a1:
            max_store = max(x3s, max_store)

print('The domain of attraction radius is {0}'.format(max_store))
# # Vdot(x)
# m_t = x1
# m_l = x2
# p_t = x3
# p_l = x4

# x1d = K * b_t/(b_t + p_l) - d_t * m_t
# x2d = K * b_l/(b_l + p_t) - d_l * m_l
# x3d = beta_t * m_t - del_t * p_t
# x4d = beta_l * m_l - del_l * p_l
# Vdx = 3.4* x1 * x1d + 6.3 * x2 * x2d + 10.4 *x3 *x3d + 105.28 *x4*x4d - 0.92 * x1d * x2 -0.92 *x1 * x2d + 3.8 *x1d * x3 + 3.8 * x1 * x3d - 18.4* x1d * x4 - 18.4 * x1 * x4d - 25* x2d * x3 - 25 * x2 * x3d + 5.28 * x2d * x4 + 5.28 * x2 * x4d - 21.76 * x3d * x4 - 21.76 * x3 * x4d
# print('The derivative of the Lyapunov function is {0}'.format(Vdx))
# x1v = np.linspace(m_te - 0.75*m_te,m_te - 0.35*m_te,10)
# x2v = np.linspace(m_le - 0.75*m_le,m_te - 0.35*m_le,10)
# x3v = np.linspace(p_te - 0.75*p_te,m_te - 0.35*p_te,10)
# x4v = np.linspace(p_le - 0.75*p_le,m_te - 0.35*p_le,10)

# for i,j,k,l in zip(x1v,x2v,x3v,x4v):
#     Vs = Vdx
#     Vs = Vs.subs(x1,i)
#     Vs = Vs.subs(x2,j)
#     Vs = Vs.subs(x3,k)
#     Vs = Vs.subs(x4,l)
    # print(Vs)
# for i in range(4):
#         plt.subplot(2,2,i+1)
#         # Plot bound

#         # Plot solutions from numerical simulation
#         plt.legend()
#         plt.xlabel('Time')
#         plt.ylabel('State x_' + str(i+1))
# plt.show()
