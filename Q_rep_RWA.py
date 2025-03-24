import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator
from Symbolic_Time_Deriv_Solver import SymbolicSolver
import sympy as sp

gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym = sp.symbols('gamma_sym eta_sym nu_sym omega_sym n_th_sym m_th_sym g_sym')
x_sym, y_sym, u_sym, v_sym, a, b, c, d, e, f, h, k, l, o, p, q, r, s, w = sp.symbols('x_sym y_sym u_sym v_sym a b c d e f h k l o p q r s w')

def solve_time_deriv_sym(symsolver, PHYS_model):  
    # Call the Fokker-Planck solver to get the time derivative functions
    TD_consts = symsolver.two_mode_fokker_planck(PHYS_model, x_sym, y_sym, u_sym, v_sym, 
                                                 a, b, c, d, e, f, h, k, l, o, p, q, r, s, w)

    # Unpack the time derivatives
    TD = [
        TD_consts[i] for i in range(15)
    ]
    
    # Create symbolic variables for the system_time_evolution
    symbols = (a, b, c, d, e, f, h, k, l, o, p, q, r, s, w, gamma_sym, eta_sym, nu_sym, omega_sym, 
               n_th_sym, m_th_sym, g_sym)

    # Create lambdified functions for all derivatives using a loop
    time_deriv_funcs = [
        sp.lambdify(symbols, TD[i], 'numpy') for i in range(15)
    ]
    
    return tuple(time_deriv_funcs)

def system_time_evolution(t, y, phys_parameter, time_deriv_funcs):
    # Unpack physical parameters
    gamma, eta, nu, omega, n_th, m_th, g= phys_parameter

    # Unpack state variables
    a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0 = y

    # Unpack the derivative functions
    derivatives = [func(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
                   for func in time_deriv_funcs]

    return np.array(derivatives)

# Initialize simulation parameters
#   Physical Parameter
gamma = 0.5
eta = 0.5
nu = 2 * np.pi
omega = 2 * np.pi
g =  2 * np.pi * 0.1
n_th = 0.01
m_th = 0.01

phys_parameter = gamma, eta, nu, omega, n_th, m_th, g

#   Initial Condition
x0 = 2
y0 = 2
u0 = 0
v0 = 0

#   Time parameter
t_start = 0
t_end = 10
dt= 0.01
simulation_time_setting = t_start, t_end, dt

#   Map Parameter
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
simulation_grid_setting = x, y

# Symbolic formula solver settings
def RWA_QREP_FKEQ(x,y,u,v,Px,Py,Pu,Pv,Pxx,Pyy,Puu,Pvv,Pxy,Puv,Pxu,Pyu,Pxv,Pyv):
    TD1 = (gamma_sym + eta_sym) + (gamma_sym/2 * x - nu_sym * y - g_sym * v) * Px + (gamma_sym/2 * y + (nu_sym) * x + g_sym * u) * Py
    TD2 = (eta_sym/2 * u - (omega_sym) * v - g_sym *y) * Pu + (eta_sym/2 * v + (omega_sym) * u + g_sym * x) * Pv
    TD3 = gamma_sym * (n_th_sym+1) / 4 * (Pxx + Pyy) + eta_sym * (m_th_sym+1) / 4 * (Puu + Pvv)
    return TD1 + TD2 + TD3  

SymSolver = SymbolicSolver()
time_deriv_funcs = solve_time_deriv_sym(SymSolver, RWA_QREP_FKEQ)
init_func_parameter = SymSolver.init_func_parameter(mode_number='two_mode',init_cond=[x0, y0, n_th,u0, v0, m_th],state_name='coherent',representation='Q')

# Instantiate and run the simulation
simulator = FokkerPlanckSimulator(representation='Q', simulation_time_setting=simulation_time_setting, simulation_grid_setting=simulation_grid_setting,\
        phys_parameter=phys_parameter, init_func_parameter=init_func_parameter, output_dir="Q_rep_RWA", probdensmap_mode='two_mode',\
        system_time_evolution=system_time_evolution, time_deriv_funcs=time_deriv_funcs)
simulator.run_simulation(pure_parameter = False)
simulator.electric_field_evolution()