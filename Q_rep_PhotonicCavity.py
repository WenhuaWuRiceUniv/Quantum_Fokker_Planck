import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator
from Symbolic_Time_Deriv_Solver import SymbolicSolver
import sympy as sp
import cmath

gamma_sym, nu_sym, n_th_sym = sp.symbols('gamma_sym nu_sym n_th_sym')
Er_sym, Ei_sym= sp.symbols('Er_sym Ei_sym')
gk_sym, freq_k_sym, Dk_sym = sp.symbols('gk_sym freq_k_sym Dk_sym')

SymSolver = SymbolicSolver(output_dir="Q_rep_PhotonicCavity")

parameter_sym = gamma_sym, nu_sym, n_th_sym, gk_sym, freq_k_sym, Dk_sym, Er_sym, Ei_sym

# Symbolic formula solver settings
def Photon_QREP_FKEQ(variable, fst_deriv, snd_deriv, mix_deriv):
    x,y = variable
    Px,Py = fst_deriv
    Pxx,Pyy = snd_deriv
    TD1 = gamma_sym + (gamma_sym / 2 * x - nu_sym * y) * Px + (gamma_sym / 2 * y + nu_sym * x) * Py
    TD2 = gamma_sym * (n_th_sym+1) / 4 * (Pxx + Pyy)
    return TD1 + TD2

def Photonic_Input(variable, fst_deriv, snd_deriv, mix_deriv):
    Px,Py = fst_deriv
    return gk_sym * Er_sym * Py + gk_sym * Ei_sym * Px

phys_model = SymSolver.build_model('one_mode', parameter_sym ,Photon_QREP_FKEQ, Photonic_Input, None)
time_deriv_funcs = SymSolver.solve_time_deriv_sym(phys_model, PHYS_model_output_file="PhotonicCavity_Model.txt")

# Initialize simulation parameters
#   Physical Parameter
gamma_value = 0.2
nu_value = 3
n_th_value = 1

#   Initial Condition
x0 = 2
y0 = 2

#   Time parameter
t_start = -10
t_end = 150
dt= 0.01
times = np.linspace(t_start,t_end,int((t_end-t_start)/dt))
simulation_time_setting = t_start, t_end, dt

#   Map Parameter
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
simulation_grid_setting = x, y

def Input_electric_field(t):
    return 10 * np.exp(-t**2/4) * np.cos(3*t) /((np.pi**0.5) *2) , 10 * np.exp(-t**2/4) * np.sin(3*t)/((np.pi**0.5) *2)

t_num = int((t_end-t_start)/dt)

gk_optical = 1/(2**0.5)
spec_freq_num = 20
spec_freq = np.linspace(0, 6, spec_freq_num)
widx = 0
Dk_value = gamma_value / (2*gk_optical**2)


# Instantiate and run the simulation
def update_phys_parameter(t):
    gamma, nu, n_th= gamma_value, nu_value, n_th_value
    Er, Ei = Input_electric_field(t)
    gk = gk_optical
    freq_k = spec_freq[widx]
    Dk = Dk_value

    return gamma, nu, n_th, gk, freq_k, Dk, Er, Ei

def system_time_evolution(t, y):
    
    # Unpack state variables
    solution_value = y

    # Unpack physical parameters
    parameter_value = update_phys_parameter(t)

    # Unpack the derivative functions
    derivatives = [func(solution_value, parameter_value)
                   for func in time_deriv_funcs]

    return np.array(derivatives)

simulator = FokkerPlanckSimulator(representation='Q', simulation_time_setting=simulation_time_setting, simulation_grid_setting=simulation_grid_setting)

unity, x, y = sp.symbols('unity x y')

def map_func(vars):
    unity, x, y= vars
    return x,y

#   Plot the snapshots of the system evolution
init_func_parameter = SymSolver.init_func_parameter(init_cond=[0, 0, n_th_value], state_name='thermal', representation='Q')
simulator.initialize(init_func_parameter=init_func_parameter)
    
# Run the simulation
simulator.run_simulation(system_time_evolution=system_time_evolution, 
                        output_dir="Q_rep_PhotonicCavity", 
                        pure_parameter=True)#False, vars_sym = [unity, x, y], Map_func = map_func)

def real_part(vars):
    unity, x, y= vars
    return x * unity
def imag_part(vars):
    unity, x, y= vars
    return y * unity

simulator.time_domain_signal(Input_electric_field, gk_optical * Dk_value, vars=[unity, x, y], real_part=real_part, imag_part=imag_part)