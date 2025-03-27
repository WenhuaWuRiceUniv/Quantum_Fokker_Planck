import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cmath
import sympy as sp
from scipy.linalg import eigh
from scipy.linalg import eig
import json

class FokkerPlanckSimulator:
    def __init__(self, representation='Q', simulation_time_setting=None, simulation_grid_setting=None):
        # Simulation time settings
        self.representation = representation
        self.t_start, self.t_end, self.dt = simulation_time_setting
        self.nsteps = int((self.t_end - self.t_start) / self.dt)
        self.time_values = np.linspace(self.t_start, self.t_end, self.nsteps)

        # Grid for 2D probability representation value
        self.x, self.y = simulation_grid_setting

        # Choose RK4 to solve for each time step
        self.solver = self.rk4_step

        # Initialize list to store centroid coordinates
        self.centroid_x = []
        self.centroid_y = []
    
    def initialize(self, init_func_parameter=None):
        # Functions for simulation
        self.mode_num =int(((8 * len(init_func_parameter) + 1)**0.5 - 3) / 4)

        # Initial conditions and solution storage
        self.init_func_parameter = init_func_parameter
        self.solution = np.zeros((self.nsteps, len(self.init_func_parameter)))
        self.solution[0] = self.init_func_parameter
        self.centroid_x = []
        self.centroid_y = []

    def run_simulation(self, system_time_evolution = None, output_dir=None, pure_parameter = False, vars_sym = None, Map_func = None, plot_centroid = True):
        self.system_time_evolution = system_time_evolution
        self.plot_centroid = plot_centroid

        # Prepare output directory
        self.output_dir = str('bin/')+output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        log_filename = os.path.join(self.output_dir, "simulation_log.txt")

        if pure_parameter:
            for t_idx in tqdm(range(1, self.nsteps), desc="Simulating", unit="step"):
                self.solution[t_idx] = self.solver(self.time_values[t_idx-1], self.solution[t_idx-1], self.dt)
        else:
            with open(log_filename, 'w') as log_file:
                log_file.write(f"Simulation started at {self.t_start} with time steps from {self.t_start} to {self.t_end}.\n")
                # Precompute min and max values for consistent color scaling in plots
                os.makedirs(self.output_dir+str('/Snapshots_')+str(Map_func(vars_sym)[0])+str(Map_func(vars_sym)[1])+str('/'), exist_ok=True)
                self.u_init = self.ProbDensMap(vars_sym, Map_func,0)
                self.vmin, self.vmax = np.min(self.u_init), np.max(self.u_init)

                log_file.write(f"Initial representation value range: vmin={self.vmin}, vmax={self.vmax}.\n")
                
                # Time integration using RK4 with progress bar
                for t_idx in tqdm(range(1, self.nsteps), desc="Simulating", unit="step"):
                    self.solution[t_idx] = self.solver(self.time_values[t_idx-1], self.solution[t_idx-1], self.dt)

                    # Compute the Representation value at the current time step
                    ProbDens = self.ProbDensMap(vars_sym, Map_func,t_idx)
                    
                    # Calculate the center of the probability distribution cloud
                    weighted_sum_x = np.sum(self.x[:, None] * ProbDens)  # Sum over x for each y
                    weighted_sum_y = np.sum(self.y[None, :] * ProbDens)  # Sum over y for each x
                    total_weight = np.sum(ProbDens)  # Total sum (normalization factor)

                    # Centroid coordinates
                    center_x = weighted_sum_x / total_weight
                    center_y = weighted_sum_y / total_weight

                    # Store the centroid coordinates
                    self.centroid_x.append(center_x)
                    self.centroid_y.append(center_y)

                    # Inside your run_simulation method (where the centroid and ProbDensSum are calculated)
                    ProbDensSum = self.volume_integration(self.x, self.y, ProbDens)

                    # Log the information and check for deviation from 1.0
                    self.log_warning_message(t_idx, ProbDensSum, center_x, center_y, log_file)

                    # Save a snapshot every 10 steps
                    if t_idx % 10 == 0:
                        self.save_snapshot(t_idx, ProbDens,f'{self.output_dir}/'+str('/Snapshots_')+str(Map_func(vars_sym)[0])+str(Map_func(vars_sym)[1])+str('/')+f'/snapshot_{t_idx:04d}.png')
                        log_file.write(f"Snapshot saved for time step {t_idx}.\n")
            
                log_file.write(f"Simulation ended at {self.t_end}.\n")
                log_file.write("Simulation completed successfully.\n")
        
    # RK4 Solver
    def rk4_step(self, t, y, dt):
        # Get the k1, k2, k3, k4 slopes
        k1 = dt* self.system_time_evolution(t, y)
        k2 = dt* self.system_time_evolution(t + dt/2, y + k1/2)
        k3 = dt* self.system_time_evolution(t + dt/2, y + k2/2)
        k4 = dt* self.system_time_evolution(t + dt, y + k3)
        
        # Update the solution using the weighted average
        return y + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    # Plotting complex representation function with parameters
    def ProbDensMap(self, vars_sym, Map_func,t_idx):
        Map_func_sym = Map_func(vars_sym)
        var_vec = self.convert_coeff_vec(vars_sym,Map_func_sym)
        var_vec[0] = 1.
        X, Y = np.meshgrid(self.x, self.y)

        A = self.characteristic_matrix(t_idx)
        indices_xy = np.where(var_vec == 1)[0]
        indices_uv = np.where(var_vec == 0)[0]
        
        if len(indices_uv)!=0:
            # Select the rows and columns from M corresponding to those indices
            Axy = np.copy(A)[indices_xy, :][:, indices_xy]
            Auv = np.copy(A)[indices_uv, :][:, indices_uv]
            Auv = np.pad(Auv, pad_width=((1, 0), (1, 0)), mode='constant', constant_values=0)

            Mix_matrix = np.copy(A)[indices_uv, :][:, indices_xy]

            D, ave_vec, a_new, P =  self.Dignolize_Characteristic_Matrix(Auv)
            result = np.pi**((self.mode_num-1))/(np.abs(np.linalg.det(D)))**0.5
            Mix_matrix = P.T @ Mix_matrix
            for i in range(len(D)):
                result = result * np.exp(-(Mix_matrix[i,0] + Mix_matrix[i,1] * X + Mix_matrix[i,2] * Y)**2/D[i,i])
            ProbDens = result * np.exp(Axy[0,0] + Axy[0,1]*2. * X + Axy[0,2]*2. * Y + Axy[1,1] * X**2 + Axy[2,2] * Y**2 + Axy[1,2]*2. * X*Y)
        else:
            ProbDens = np.exp(A[0,0] + A[0,1] * 2 * X + A[0,2] * 2 * Y + A[1,1] * X**2 + A[2,2] * Y**2 + A[1,2]*2. * X*Y)
    
        return ProbDens
        
    def Dignolize_Characteristic_Matrix(self,A):
        A_new = np.copy(A[1:,1:])
        # Diagonalize the matrix
        eigvals, eigvecs = eigh(A_new)
        # Create the diagonal matrix with eigenvalues
        D = np.diag(eigvals)      
        P = eigvecs
            
        # Eliminate 1st order terms
        u_vec = np.ones(len(D)) @ np.diag(A[0,1:]*2.) @ P
        a_new = A[0,0]
        ave_vec = []
        for i in range(len(D)):
            a_new = a_new - u_vec[i] **2 / 4. / D[i,i]
            ave_vec.append(-u_vec[i] / 2. / D[i,i])
            if D[i,i]>0:
                print('Error: Non-negative diagonal value!')
        return D, np.array(ave_vec), a_new, P

    def characteristic_matrix(self, t_idx):

        sol_value = self.solution[t_idx]
        A = np.zeros((2 * self.mode_num + 1,2 * self.mode_num + 1))
        A[0,0] = sol_value[0]

        for i in range(2 * self.mode_num):
            A[i+1,0] = sol_value[1+i] / 2.
            A[0,i+1] = sol_value[1+i] / 2.

        for i in range(2 * self.mode_num):
            A[i+1,i+1] = sol_value[1+2*self.mode_num+i]
        
        num = 0
        for i in range(2 * self.mode_num):
            for j in range(i+1,2 * self.mode_num):
                A[i+1,j+1] = sol_value[1+4*self.mode_num+num]/2.
                A[j+1,i+1] = sol_value[1+4*self.mode_num+num]/2.
                num = num + 1

        return A
    
    def Wehr_Entropy(self,t_idx):
        A = self.characteristic_matrix(t_idx)
        D, ave_vec, a_new, P = self.Dignolize_Characteristic_Matrix(A)
        result = 1
        for i in range(len(D)):
            result = result * (np.pi / (-D[i,i]))**0.5
        return - (a_new + 1/2. * len(D)) * np.exp(a_new) *result
    
    def Wehr_Entropy_all_time(self):
        entropies = []
        for t_idx in tqdm(range(len(self.time_values)), desc="Computing Expectation Values"):
            entropies.append(self.Wehr_Entropy(t_idx))

        return np.array(self.time_values), np.array(entropies)
      
    def convert_coeff_vec(self, vars_sym, Func_sym):
        coeff_vec = np.zeros(len(vars_sym))
        for i in range(len(vars_sym)):
            for j in range(len(Func_sym)):
                coeff_vec[i] = coeff_vec[i] + (Func_sym[j].coeff(vars_sym[i],1))
        coeff_vec = np.array(coeff_vec)
        return coeff_vec
    
    def convert_coeff_matrix(self, vars_sym, Func_sym):
        coeff_matrix = []
        for i in range(len(vars_sym)):
            temp = []
            for j in range(len(vars_sym)):
                if i>j:
                    temp.append(Func_sym.coeff(vars_sym[i],1).coeff(vars_sym[j],1))
                elif i==j:
                    temp.append(Func_sym.coeff(vars_sym[i],2))
                else:
                    temp.append(0)
            coeff_matrix.append(temp)
        coeff_matrix = np.array(coeff_matrix)
        return coeff_matrix
    
    def expectation_value(self, coeff_matrix, t_idx):
        A = self.characteristic_matrix(t_idx)
        D, ave_vec, a_new, P = self.Dignolize_Characteristic_Matrix(A)

        # Compute determinant factor efficiently
        result = np.prod(np.sqrt(np.pi / -np.diag(D)))

        U0 = coeff_matrix[0, 0]
        ua = coeff_matrix[0, 1:] + coeff_matrix[1:, 0].T
        UB = coeff_matrix[1:, 1:]

        # Faster transformation without explicit inversion
        converted_coeff = P.T @ UB @ P  

        # Optimized calculations
        PA_ave_vec = np.einsum('i,ij->j', ua, P) @ ave_vec  
        coeff = U0 + np.sum(PA_ave_vec) + np.einsum('i,ij,j->', ave_vec, converted_coeff, ave_vec)

        # Efficient diagonal sum
        coeff += np.sum(np.diag(converted_coeff) / (-2 * np.diag(D)))

        return result * np.exp(a_new) * coeff

    def expectation_value_alltime(self, vars_sym, Exp_Func):
        Func_sym = sp.expand(Exp_Func(vars_sym))  
        coeff_matrix = self.convert_coeff_matrix(vars_sym, Func_sym)

        # Precompute values using parallel processing
        time_indices = np.arange(len(self.time_values))
        values = np.array([self.expectation_value(coeff_matrix, t_idx) for t_idx in tqdm(time_indices, desc="Computing Expectation Values")])

        return np.array(self.time_values), values
    
    def expectation_value_final(self, vars_sym, Exp_Func):
        Func_sym = sp.expand(Exp_Func(vars_sym))  
        coeff_matrix = self.convert_coeff_matrix(vars_sym, Func_sym)
        value = self.expectation_value(coeff_matrix,-1)

        return value
        
    def volume_integration(self, x, y, Dens):
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        Sum = np.sum(Dens) * np.abs(dx) * np.abs(dy)
        return Sum
    
    def rms_error(self, X,Y):
        return (np.sum((X - Y)**2)/np.sum(X)**2)**0.5
    
    def log_warning_message(self, t_idx, ProbDensSum, center_x, center_y, log_file):
        # Calculate the difference from 1.0
        diff_from_one = abs(ProbDensSum - 1.0)

        # Check if the difference is more than 1% away from 1.0
        if diff_from_one > 0.01:
            log_file.write(f"WARNING: Step {t_idx}, Time = {self.time_values[t_idx]:.2f} ps, ProbDensSum = {ProbDensSum:.4f}, Center = ({center_x:.4f}, {center_y:.4f})\nProbability represenation value summation on the plane has deviation larger than 1.0%! Simulation Box may be too small.\n")
        else:
            log_file.write(f"Step {t_idx}, Time = {self.time_values[t_idx]:.2f} ps, ProbDensSum = {ProbDensSum:.4f}, Center = ({center_x:.4f}, {center_y:.4f})\n")
    
    def save_snapshot(self, t_idx, ProbDens, name):
        # Create a plot for the Represenation Value
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(ProbDens.T, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin='lower', aspect='auto', cmap='hot')#, vmin=self.vmin, vmax=self.vmax)
        ax.set_title(self.representation + f"-representation Time-evolution at Time = {self.time_values[t_idx]:.2f} ps")
        ax.set_xlabel(r'Re{$\alpha$} (x)')
        ax.set_ylabel(r'Im{$\alpha$} (y)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Representation Value')

        if self.plot_centroid:
            # Plot the path of the center
            ax.plot(self.centroid_x, self.centroid_y, 'w-', label='Center Path', linewidth=2)
            ax.legend(loc="upper right")

        # Save snapshot
        plt.savefig(name)
        plt.close(fig)

    def exctract_expectation_value(self, vars, real_part, imag_part, filename):
        expectation_file = f'{self.output_dir}/'+filename+"field_amplitude_exp_value.json"
        
        if os.path.exists(expectation_file):
            print("Loading previously computed expectation values...")
            with open(expectation_file, "r") as f:
                expectation_values = json.load(f)
            ts = np.array(expectation_values["time"])
            x_value = np.array(expectation_values["real_part"])
            y_value = np.array(expectation_values["imag_part"])
        else:
            print('Extracting Expectation Value')
            ts, x_value = self.expectation_value_alltime(vars, real_part)
            ts, y_value = self.expectation_value_alltime(vars, imag_part)
            print('Expectation Value Extracted')

            # Convert sympy objects to float
            ts_list = [float(t) for t in ts]
            x_value_list = [float(x) for x in x_value]
            y_value_list = [float(y) for y in y_value]

            # Save expectation values
            expectation_values = {
                "time": ts_list,
                "real_part": x_value_list,
                "imag_part": y_value_list
            }
            with open(expectation_file, "w") as f:
                json.dump(expectation_values, f, indent=4)
        return ts, x_value, y_value
    
    def time_domain_signal(self, input_field, gk_Dk_optical, ts, x_value, y_value, if_plot):
        
        input_field_amp = [input_field(ts[i])[0] for i in range(len(ts))]
        transmission = gk_Dk_optical * y_value
        reflection = np.array(input_field_amp) + gk_Dk_optical * y_value
        cavity_photon_num = x_value**2 + y_value**2

        dt = ts[1] - ts[0]
        ref_spec = np.fft.fft(reflection) * dt
        tra_spec = np.fft.fft(transmission) * dt
        frequency = np.fft.fftfreq(len(ref_spec), dt)  # Frequency bins

        if if_plot:
            # Plot and save observed electric field signal in time domain
            plt.plot(ts, cavity_photon_num, label='Cavity Photon Number')
            plt.xlim([ts[0], ts[int(len(ts)/1.1)]])
            plt.legend()
            plt.xlabel(r'time(ps)')
            plt.ylabel(r'Cavity Photon Number')
            plt.savefig(f'{self.output_dir}/'+"cavity_photon_number.png")
            
            plt.plot(ts, input_field_amp, label='Input Field')
            plt.plot(ts, transmission, label='Transmission')
            plt.plot(ts, reflection, label='Reflection')
            plt.xlim([ts[0], ts[int(len(ts)/1.1)]])
            plt.legend()
            plt.xlabel(r'time(ps)')
            plt.ylabel(r'Amplitude')
            plt.savefig(f'{self.output_dir}/'+"time_domain_signal.png")

            # Plot and save the FFT spectrum
            

            #plt.plot(frequency, np.abs(ref_spec)**2, label='Reflection')
            plt.plot(frequency, np.abs(tra_spec)**2, label='Transmission')
            #plt.plot(frequency, np.abs(input_spec)**2, label='Input Field')
            plt.xlim([0, 1])
            plt.legend()
            plt.xlabel(r'freq(THz)')
            plt.ylabel(r'Amplitude')
            plt.savefig(f'{self.output_dir}/'+"fft_spectrum.png")
        return ts, cavity_photon_num, input_field_amp, transmission, reflection, frequency, tra_spec, ref_spec
    
    def plot_comparison(self, ts, cavity_photon_nums, transmissions, frequency, tra_specs, identifier='wmod_1THz_', pulse = None):
    # Plot and save observed electric field signal in time domain
            plt.plot(ts, cavity_photon_nums[0], label='No mod')
            plt.plot(ts, cavity_photon_nums[1], label='Mod')
            plt.xlim([ts[0], ts[int(len(ts)/1.1)]])
            plt.legend()
            plt.xlabel(r'time(ps)')
            plt.ylabel(r'Cavity Photon Number')
            plt.savefig(f'{self.output_dir}/'+identifier+"cavity_photon_number.png")
            plt.close()

            plt.plot(ts, transmissions[0], label='No mod')
            plt.plot(ts, transmissions[1], label='Mod')
            if pulse is not None:
                plt.plot(ts, pulse/np.max(pulse)*np.max(transmissions[0]), label='Pulse')
            plt.xlim([ts[0], ts[int(len(ts)/1.1)]])
            plt.legend()
            plt.xlabel(r'time(ps)')
            plt.ylabel(r'Amplitude')
            plt.savefig(f'{self.output_dir}/'+identifier+"time_domain_signal.png")
            plt.close()

            # Plot and save the FFT spectrum
            plt.plot(frequency, np.abs(tra_specs[0])**2, label='No mod')
            plt.plot(frequency, np.abs(tra_specs[1])**2, label='Mod')

            plt.xlim([0, 1])
            plt.legend()
            plt.xlabel(r'freq(THz)')
            plt.ylabel(r'Amplitude')
            plt.savefig(f'{self.output_dir}/'+identifier+"fft_spectrum.png")
            plt.close()
    
    def plot_centroid_path(self):
        ts = []
        for t_idx in range(1, self.nsteps):
            ts.append(self.time_values[t_idx])
        plt.plot(ts,self.centroid_x,label="cavity photon field amp real part")
        plt.plot(ts,self.centroid_y,label="cavity photon field amp imag part")
        plt.xlabel(r'time(ps)')
        plt.ylabel(r'Amplitude')
        plt.legend()
        plt.show()