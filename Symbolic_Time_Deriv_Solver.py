import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cmath
import sympy as sp
from itertools import combinations
import sys
import scipy.fft as fft

class SymbolicSolver:
    def __init__(self, output_dir):
        # Simulation time settings
        self.var = 0
        # Prepare output directory
        self.output_dir = str('bin/')+output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    # 一階導數
    def fst_deriv(self, func, var):
        return sp.simplify(sp.diff(func, var))

    # 二階導數
    def snd_deriv(self, func, var):
        return sp.simplify(sp.diff(func, var, var))

    # 混合二階導數
    def mix_deriv(self, func, var1, var2):
        return sp.simplify(sp.diff(sp.diff(func, var1), var2))

    # 指數函數的一階導數
    def exp_fst_deriv(self, func, var):
        return sp.simplify(sp.diff(func, var) / func)

    # 指數函數的二階導數
    def exp_snd_deriv(self, func, var):
        return sp.simplify(sp.diff(sp.diff(func, var), var) / func)

    # 指數函數的混合二階導數
    def exp_mix_deriv(self, func, var1, var2):
        return sp.simplify(sp.diff(sp.diff(func, var1), var2) / func)
    
    def generate_sol_sym(self, mode_num):
        if mode_num=='one_mode':
            a, b, c, d, e, f = sp.symbols('a b c d e f')
            return a, b, c, d, e, f
        elif mode_num=='two_mode':
            a, b, c, d, e, f, h, k, l, o, p, q, r, s, w = sp.symbols('a b c d e f h k l o p q r s w')
            return a, b, c, d, e, f, h, k, l, o, p, q, r, s, w
        elif mode_num=='three_mode':
            a, b, c, d, e, f, h, k, l, o, p, q, r, s, w, ku, ke, ko, sa, si, su, se, so, ta, ti, tu, te, to =\
                sp.symbols('a b c d e f h k l o p q r s w ku ke ko sa si su se so ta ti tu te to')
            return a, b, c, d, e, f, h, k, l, o, p, q, r, s, w, ku, ke, ko, sa, si, su, se, so, ta, ti, tu, te, to
    
    def generate_var_sym(self, mode_num):
        if mode_num=='one_mode':
            x_sym, y_sym = sp.symbols('x_sym y_sym')
            return x_sym, y_sym
        elif mode_num=='two_mode':
            x_sym, y_sym, u_sym, v_sym = sp.symbols('x_sym y_sym u_sym v_sym')
            return x_sym, y_sym, u_sym, v_sym
        elif mode_num=='three_mode':
            x_sym, y_sym, u_sym, v_sym, ka_sym, ki_sym = sp.symbols('x_sym y_sym u_sym v_sym ka_sym ki_sym')
            return x_sym, y_sym, u_sym, v_sym, ka_sym, ki_sym
    
    def build_model(self,mode_num, parameter_sym, system_hamiltonian, input_hamiltonian, output_hamiltonian):
        self.mode_num = mode_num
        self.solution_sym = self.generate_sol_sym(mode_num)
        self.variable_sym = self.generate_var_sym(mode_num)
        self.parameter_sym = parameter_sym

        phys_model = system_hamiltonian
        def add_functions(f, g):
            return lambda *args, **kwargs: f(*args, **kwargs) + g(*args, **kwargs)
        
        if input_hamiltonian is not None:
            phys_model = add_functions(phys_model, input_hamiltonian)
        
        if output_hamiltonian is not None:
            phys_model = add_functions(phys_model, output_hamiltonian)
        
        return phys_model
    
    def create_phys_model_sym(self, PHYS_model):
        var_sym, sol_sym, para_sym = self.variable_sym, self.solution_sym, self.parameter_sym  

        # Unpack the time derivatives
        fake_var = []
        for i in range(len(var_sym)):
            fake_var.append(sp.symbols(f'fake_var_{i}'))

        fake_fst_deriv = []
        for i in range(len(var_sym)):
            fake_fst_deriv.append(sp.symbols(f'fake_fst_deriv_{i}'))

        fake_snd_deriv = []
        for i in range(len(var_sym)):
            fake_snd_deriv.append(sp.symbols(f'fake_snd_deriv_{i}'))

        fake_mix_deriv = []
        mix_dim = int(len(var_sym)*(len(var_sym)-1)/2)
        for i in range(mix_dim):
            fake_mix_deriv.append(sp.symbols(f'fake_mix_deriv_{i}'))

        Phys_model_sym = sp.expand(PHYS_model(fake_var, fake_fst_deriv, fake_snd_deriv, fake_mix_deriv))
        return Phys_model_sym
        

    def solve_time_deriv_sym(self, PHYS_model, PHYS_model_output_file):
        var_sym, sol_sym, para_sym = self.variable_sym, self.solution_sym, self.parameter_sym

        # Create a symbol for the physics model
        Phys_model_sym = self.create_phys_model_sym(PHYS_model)

        # Load previous results, if there is any
        TD = self.load_time_deriv_sym(expected_phys_model=Phys_model_sym, PHYS_model_output_file=PHYS_model_output_file)

        if len(TD)==0:
            # Call the Fokker-Planck solver to get the time derivative functions
            TD_consts = self.time_deriv_fokker_planck(PHYS_model, var_sym, sol_sym)      

            # Unpack the time derivatives
            TD = [TD_consts[0]]
            for i in range(len(var_sym)):
                TD.append(TD_consts[1][i])
            for i in range(len(var_sym)):
                TD.append(TD_consts[2][i])
            mix_dim = int(len(var_sym)*(len(var_sym)-1)/2)
            for i in range(mix_dim):
                TD.append(TD_consts[3][i])
        
        # Create symbolic variables for the system_time_evolution
        symbols = (sol_sym, para_sym)

        # Create lambdified functions for all derivatives using a loop
        time_deriv_funcs = [
            sp.lambdify(symbols, TD[i], 'numpy') for i in range(len(sol_sym))
        ]

        self.output_time_deriv_sym(Phys_model_sym,TD, PHYS_model_output_file)

        return tuple(time_deriv_funcs)
    
    def output_time_deriv_sym(self,Phys_model_sym,TD, PHYS_model_output_file):
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Define the file path
        file_path = os.path.join(self.output_dir, PHYS_model_output_file)

        # Save the symbolic expressions using srepr() to store them as strings
        with open(file_path, 'w') as f:
            f.write(f"Physics Model: {sp.srepr(Phys_model_sym)}\n")
            for i in range(len(TD)):
                f.write(f"TD[{i}]: {sp.srepr(TD[i])}\n")

        print(f"Symbolic time-derivative functions saved to {file_path}")

    def load_time_deriv_sym(self, expected_phys_model, PHYS_model_output_file):
        file_path = os.path.join(self.output_dir, PHYS_model_output_file)
        TD = []
        # Check if the file exists
        if not os.path.exists(file_path):
            print("No previous results! Need to calculate the time-derivative functions.")
            return TD  # Return empty list or handle as needed
        
        with open(file_path, 'r') as f:
            # Read the first line (Physics Model)
            lines = f.readlines()
            first_line = lines[0].strip()
            
            # Extract the Physics Model from the first line
            phys_model_str = first_line.split(":")[1].strip()
            phys_model = sp.sympify(phys_model_str)
            
            # Compare with the expected symbolic object
            if phys_model != expected_phys_model:
                print(f"Physics model changes! Need to recalculate the time-derivative functions")

            # If the comparison is successful, load the time derivatives
            else:
                print(f"Physics model remains the same! Just load the previous results")
                for line in lines[1:]:  # Start from the second line (TD expressions)
                    if line.startswith("TD["):
                        # Use sympy.sympify to convert the string back into a symbolic expression
                        sym_expr = sp.sympify(line.split(":")[1].strip())
                        TD.append(sym_expr)
                    
            return TD
    
    def generate_Ef_Et(self, ws,times,Input_E_field_spec):
        ang_freq = np.array(ws)
        Efs = Input_E_field_spec(ang_freq)
        dw = ang_freq[1] - ang_freq[0]
        Ets = []
        for t in range(len(times)):
            Et = 0
            for w in range(len(ang_freq)):
                Et += Efs[w] * dw * np.exp(1j * ang_freq[w] * times[t])/(2*np.pi)
            Ets.append(Et)
        Ets = np.array(Ets)
        return Efs.real, Efs.imag, Ets.real, Ets.imag
    
    def input_freq_amplitude(self, freq):

        amplitude_re_input = np.interp(freq, self.ang_freq, self.Re_E_f)
        amplitude_im_input = np.interp(freq, self.ang_freq, self.Im_E_f)

        return amplitude_re_input, amplitude_im_input
    
    def init_func_parameter(self, init_cond=None, state_name='coherent', representation='Q'):
        dim = int(len(init_cond)/3)

        eps2 = []
        for i in range(dim):
            if representation == 'P':
                if state_name == 'coherent':
                    eps2.append(0.1)
                elif state_name == 'thermal':
                    eps2.append(init_cond[i * 3 + 2])
            elif representation == 'Q':
                if state_name == 'coherent':
                    eps2.append(1)
                elif state_name == 'thermal':
                    eps2.append(init_cond[i * 3 + 2] + 1)

        init_func_parameter = []

        a = 0
        for i in range(dim):
            a = a - (init_cond[i * 3 + 0] ** 2 + init_cond[i * 3 + 1] ** 2)/eps2[i] - 2 * np.log( np.sqrt(np.pi * eps2[i]))
        init_func_parameter.append(a)

        for i in range(dim):
            init_func_parameter.append(2 * init_cond[i * 3 + 0]/eps2[i])
            init_func_parameter.append(2 * init_cond[i * 3 + 1]/eps2[i])
        for i in range(dim):
            init_func_parameter.append(-1/eps2[i])
            init_func_parameter.append(-1/eps2[i])
        pair_dim = int(dim*2 * (dim*2-1) / 2)
        for i in range(pair_dim):
            init_func_parameter.append(0)

        return init_func_parameter
    

    def time_deriv_fokker_planck(self, PHYS_model, variable, solution):
        print("Calculating time derivatives using Fokker-Planck solver...")
        sys.stdout.flush()  # Ensure the print statement is immediately displayed

        # Get all unique pairs (combinations of 2)
        pairs = list(combinations(variable, 2))

        # Generate the product of each pair
        products = [pair[0] * pair[1] for pair in pairs]

        func = sp.exp(solution[0])
        for i in tqdm(range(len(variable)), desc="Processing exp(solution[1+i] * variable[i])"):
            func *= sp.exp(solution[1 + i] * variable[i])
        
        for i in tqdm(range(len(variable)), desc="Processing exp(solution[1+len(variable)+i] * variable[i]**2)"):
            func *= sp.exp(solution[1 + len(variable) + i] * variable[i]**2)
        
        for i in tqdm(range(len(products)), desc="Processing exp(solution[1+len(variable)*2+i] * products[i])"):
            func *= sp.exp(solution[1 + len(variable) * 2 + i] * products[i])

        # Now, perform the first derivative calculations
        F_fst_deriv = []
        for i in tqdm(range(len(variable)), desc="Calculating first derivatives"):
            F_fst_deriv.append(self.exp_fst_deriv(func, variable[i]))
        
        # Now, perform the second derivative calculations
        F_snd_deriv = []
        for i in tqdm(range(len(variable)), desc="Calculating second derivatives"):
            F_snd_deriv.append(self.exp_snd_deriv(func, variable[i]))

        # Now, perform the mixed derivative calculations
        F_mix_deriv = []
        for i in tqdm(range(len(pairs)), desc="Calculating mixed derivatives"):
            F_mix_deriv.append(self.exp_mix_deriv(func, pairs[i][0], pairs[i][1]))

        # Calculate the time derivative of the physical model
        Phys_model_sym = sp.expand(PHYS_model(variable, F_fst_deriv, F_snd_deriv, F_mix_deriv))

        # Second derivative sum
        Snd_Deriv_Sum = 0
        TD_snd_deriv = []
        for i in tqdm(range(len(variable)), desc="Processing second derivative terms"):
            TD_snd_deriv.append(Phys_model_sym.coeff(variable[i], 2))
            Snd_Deriv_Sum += TD_snd_deriv[i] * variable[i] ** 2

        # Mixed derivatives sum
        TD_mix_deriv = []
        Mix_Derivs_Sum = 0
        for i in tqdm(range(len(pairs)), desc="Processing mixed derivative terms"):
            TD_mix_deriv.append(Phys_model_sym.coeff(pairs[i][0], 1).coeff(pairs[i][1], 1))
            Mix_Derivs_Sum += TD_mix_deriv[i] * pairs[i][0] * pairs[i][1]
        Mix_Derivs_Sum = sp.expand(Mix_Derivs_Sum)

        # First derivative sum
        Fst_Deriv_Sum = 0
        TD_fst_deriv = []
        for i in tqdm(range(len(variable)), desc="Processing first derivative terms"):
            TD_fst_deriv.append(sp.simplify(Phys_model_sym.coeff(variable[i], 1) - Mix_Derivs_Sum.coeff(variable[i], 1)))
            Fst_Deriv_Sum += TD_fst_deriv[i] * variable[i]

        # Final time derivative constant
        TD_const = sp.simplify(Phys_model_sym - Fst_Deriv_Sum - Snd_Deriv_Sum - Mix_Derivs_Sum)

        return TD_const, TD_fst_deriv, TD_snd_deriv, TD_mix_deriv
