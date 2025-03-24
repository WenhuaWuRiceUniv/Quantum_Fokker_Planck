import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator
from Symbolic_Time_Deriv_Solver import SymbolicSolver
import sympy as sp
import cmath

gamma_sym, nu_sym, n_th_sym = sp.symbols('gamma_sym nu_sym n_th_sym')
eta_sym, omega_sym, m_th_sym = sp.symbols('eta_sym omega_sym m_th_sym')
g_sym, Dia_sym = sp.symbols('g_sym Dia_sym')
Er_sym, Ei_sym= sp.symbols('Er_sym Ei_sym')
gk_input_sym, gk_output_sym, freq_k_sym, Dk_sym = sp.symbols('gk_input_sym gk_output_sym freq_k_sym Dk_sym')

SymSolver = SymbolicSolver(output_dir="Q_rep_VacuumPhoton_pulse")

parameter_sym = gamma_sym, nu_sym, n_th_sym, eta_sym, omega_sym, m_th_sym, g_sym, Dia_sym, gk_input_sym, gk_output_sym, freq_k_sym, Dk_sym, Er_sym, Ei_sym

# Symbolic formula solver settings
def VacuumPhoton_pulse_QREP_FKEQ(variable, fst_deriv, snd_deriv, mix_deriv):
    x,y,u,v = variable
    Px,Py,Pu,Pv = fst_deriv
    Pxx,Pyy,Puu,Pvv = snd_deriv
    Pxy, Pxu, Pxv, Pyu, Pyv, Puv =  mix_deriv
    TD1 = (gamma_sym + eta_sym) + (gamma_sym/2 * x - nu_sym * y) * Px + (gamma_sym/2 * y + (nu_sym + Dia_sym * 4) * x + 2* g_sym * u) * Py
    TD2 = (eta_sym/2 * u - (omega_sym + Dia_sym * 2) * v ) * Pu + (eta_sym/2 * v + (omega_sym + Dia_sym * 2) * u + 2* g_sym * x) * Pv
    TD3 = gamma_sym * (n_th_sym+1) / 4 * (Pxx + Pyy) + eta_sym * (m_th_sym+1) / 4 * (Puu + Pvv)
    TD4 = - g_sym * Pxv / 2 - g_sym * Pyu /2 - Dia_sym * Pxy
    return TD1 + TD2 + TD3 + TD4
 


def Photonic_Input(variable, fst_deriv, snd_deriv, mix_deriv):
    x,y,u,v = variable
    Px,Py,Pu,Pv = fst_deriv
    Pxx,Pyy,Puu,Pvv = snd_deriv
    return gk_input_sym * Er_sym * Py + gk_input_sym * Ei_sym * Px

phys_model = SymSolver.build_model('two_mode', parameter_sym ,VacuumPhoton_pulse_QREP_FKEQ, Photonic_Input, None)
time_deriv_funcs = SymSolver.solve_time_deriv_sym(phys_model, PHYS_model_output_file="VacuumPhoton_pulse_Model.txt")

# Initialize simulation parameters
#   Physical Parameter
gamma_value = 0.026*2*np.pi
nu_value =  np.pi
n_th_value = 0

eta_value = 0.01*2*np.pi
omega_value = np.pi
m_th_value = 0

#   Time parameter
t_start = -10
t_end = 0.5
dt= 0.001
times = np.linspace(t_start,t_end,int((t_end-t_start)/dt))
simulation_time_setting = t_start, t_end, dt

#   Map Parameter
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
simulation_grid_setting = x, y

def Input_electric_field(t):
    return 0, 0

t_num = int((t_end-t_start)/dt)

gk_optical = (gamma_value/2)**(0.5)
gk_matter = (eta_value)**0.5
spec_freq_num = 200
spec_freq = np.linspace(0, 6, spec_freq_num)
widx = 0
Dk_value = gamma_value / (2*gk_optical**2)

turn_on_mod = True
mod_strength = 20
mod_freq = 120
t_p = 0.05
t_cent = 0.

def pulse(t):
    return mod_strength * np.cos(mod_freq*t) * np.exp(-(t-t_cent)**2/2/t_p**2)
    
# Instantiate and run the simulation
def update_phys_parameter(t):
    gamma, nu, n_th= gamma_value, nu_value, n_th_value
    eta, omega, m_th= eta_value, omega_value, m_th_value
    g = 0.5

    if turn_on_mod:
        omega = omega * (1+pulse(t))
        g = g * np.sqrt(np.abs(1+pulse(t)))

    Dia = g**2/omega
    Er, Ei = Input_electric_field(t)
    gk_input = gk_optical
    gk_output = gk_optical
    freq_k = spec_freq[widx]
    Dk = Dk_value


    return gamma, nu, n_th, eta, omega, m_th, g, Dia, gk_input, gk_output, freq_k, Dk, Er, Ei

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

unity, x, y, u, v= sp.symbols('unity x y u v')
vars = [unity, x, y, u, v]
    
def map_func_photon(vars):
    unity, x, y, u, v= vars
    return x,y
def map_func_matter(vars):
    unity, x, y, u, v= vars
    return u,v

#   pulse
pulse_field = pulse(times)

#   Plot the snapshots of the system evolution
init_func_parameter = SymSolver.init_func_parameter(init_cond=[0, 0, n_th_value, 0, 0, m_th_value], state_name='thermal', representation='Q')
simulator.initialize(init_func_parameter=init_func_parameter)
    
# Run the simulation
simulator.run_simulation(system_time_evolution=system_time_evolution, 
                        output_dir="Q_rep_VacuumPhoton_pulse", 
                        pure_parameter=False, vars_sym = [unity, x, y, u, v], Map_func = map_func_photon)
time, entropy = simulator.Wehr_Entropy_all_time()
plt.plot(time, entropy)
plt.show()
"""simulator.plot_centroid_path()

def cavity_photon_number(vars):
    unity, x, y, u, v= vars
    return x*x + y*y - unity*unity
ts, cavity_photon = simulator.expectation_value_alltime(vars, cavity_photon_number)
# Plot cavity photon number on the left y-axis
fig, ax1 = plt.subplots()
ax1.plot(ts, cavity_photon, label="Cavity Photon Number", color='b')
ax1.set_xlabel(r'Time (ps)')
ax1.set_ylabel(r'Photon Number', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for pulse field strength
ax2 = ax1.twinx()
ax2.plot(ts, pulse_field , 
         label="Pulse", color='r', linestyle='dashed')
ax2.set_ylabel(r'Pulse Field Strength (T)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legends
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.show()"""