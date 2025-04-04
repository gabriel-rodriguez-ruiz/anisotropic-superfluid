# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:47:09 2024

@author: Gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
from analytic_energy import GetAnalyticEnergies

def get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda):
    """ Periodic Hamiltonian in x and y with flux.
    """
    H = (
        -2*w_0*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
        + 2*Lambda*(np.sin(k_x)*np.cos(phi_x) * np.kron(tau_z, sigma_y)
                    + np.cos(k_x)*np.sin(phi_x) * np.kron(tau_0, sigma_y)
                    - np.sin(k_y)*np.cos(phi_y) * np.kron(tau_z, sigma_x)
                    - np.cos(k_y)*np.sin(phi_y) * np.kron(tau_0, sigma_x))
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
        + Delta*np.kron(tau_x, sigma_0)
            ) * 1/2
    return H

# def get_Energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda):
#     energies = np.zeros((len(k_x_values), len(k_y_values),
#                         len(phi_x_values), len(phi_y_values), 4))
#     for i, k_x in enumerate(k_x_values):
#         for j, k_y in enumerate(k_y_values):
#             for k, phi_x in enumerate(phi_x_values):
#                 for l, phi_y in enumerate(phi_y_values):
#                     chi_k = -2*w_0*(np.cos(k_x + phi_x) + np.cos(k_y + phi_y))
#                     m = 0
#                     for s in [-1, 1]:
#                         for sj in [-1, 1]:
#                             energies[i, j, k, l, m] = s*B_y + sj*np.sqrt( (chi_k-mu)**2 + Delta**2)
#                             m += 1
#     return energies

def get_Energy_without_SOC(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y):
    energies = np.zeros((len(k_x_values), len(k_y_values),
                        len(phi_x_values), len(phi_y_values), 4))
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for k, phi_x in enumerate(phi_x_values):
                for l, phi_y in enumerate(phi_y_values):
                    a = -2*w_0 * (np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
                    b = 2*w_0 * (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
                    B_square = B_x**2 + B_y**2 
                    energies[i, j, k, l, 0] = 1/2 * (b - np.sqrt(B_square + Delta**2 - 2*np.sqrt(B_square * ((a-mu)**2 + Delta**2)) + (a-mu)**2))
                    energies[i, j, k, l, 1] = 1/2 * (b + np.sqrt(B_square + Delta**2 - 2*np.sqrt(B_square * ((a-mu)**2 + Delta**2)) + (a-mu)**2))
                    energies[i, j, k, l, 2] = 1/2 * (b - np.sqrt(B_square + Delta**2 + 2*np.sqrt(B_square * ((a-mu)**2 + Delta**2)) + (a-mu)**2))
                    energies[i, j, k, l, 3] = 1/2 * (b + np.sqrt(B_square + Delta**2 + 2*np.sqrt(B_square * ((a-mu)**2 + Delta**2)) + (a-mu)**2))
    return energies

def get_superconducting_density(L_x, L_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D, h):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    phi_x_values = [-h, 0, h]
    phi_y_values = [-h, 0, h]
    E = GetAnalyticEnergies(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D)
    negative_energy = np.where(E<0, E, 0)
    fundamental_energy = 1/2*np.sum(negative_energy, axis=(0, 1, 4))
    n_s_xx = 1/w_0 * 1/(L_x*L_y) * ( fundamental_energy[2,1] - 2*fundamental_energy[1,1] + fundamental_energy[0,1]) / h**2
    n_s_yy = 1/w_0 * 1/(L_x*L_y) * ( fundamental_energy[1,2] - 2*fundamental_energy[1,1] + fundamental_energy[1,0]) / h**2
    n_s_xy = 1/w_0 * 1/(L_x*L_y) * ( fundamental_energy[2,2] - fundamental_energy[2,0] - fundamental_energy[0,2] + fundamental_energy[0,0]) / (2*h)**2  #Finite difference of mixed derivatives
    n_s_yx = 1/w_0 * 1/(L_x*L_y) * ( fundamental_energy[2,2] - fundamental_energy[0,2] - fundamental_energy[2,0] + fundamental_energy[0,0]) / (2*h)**2
    return n_s_xx, n_s_yy, n_s_xy, n_s_yx

def get_Green_function(omega, k_x_values, k_y_values, w_0, mu, Delta, B_x, B_y, Lambda):
    G = np.zeros((len(k_x_values), len(k_y_values),
                  4, 4), dtype=complex)
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for l in range(4):
                for m in range(4):
                    G[i, j, l, m] = np.linalg.inv(omega*np.kron(tau_0, sigma_0)
                                                  - get_Hamiltonian(k_x, k_y, 0, 0, w_0, mu, Delta, B_x, B_y, Lambda) 
                                                  )[l, m]                
    return G

def get_DOS(omega, eta, L_x, L_y, w_0, mu, Delta, B_x, B_y, Lambda):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    G = get_Green_function(omega+1j*eta, k_x_values, k_y_values, w_0, mu, Delta, B_x, B_y, Lambda)
    return 1/(L_x*L_y) * 1/np.pi*np.sum(-np.imag(G), axis=(0,1))

def integrate(B):
    n = np.zeros(4)
    B_x = B * ( g_xx * np.cos(theta) + g_xy * np.sin(theta) ) 
    B_y = B * ( g_yx * np.cos(theta) + g_yy * np.sin(theta) )
    # B_x = B * np.cos(theta)
    # B_y = B * np.sin(theta)
    n[0], n[1], n[2], n[3] = get_superconducting_density(L_x, L_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D, h)
    return n

L_x = 1000#1500
L_y = 1000#1500
w_0 = 25#100 # meV
Delta = 0.2 # meV  0.2 ###############Normal state
mu = -1.98*w_0  #-3.49*w_0 	#2*(20*Delta-2*w_0)

theta = np.pi/2   #np.pi/2
a = 3.08e-07 * np.sqrt(4)#3.08e-07 # cm
g = 10
mu_B = 5.79e-2 # meV/T
Lambda_R = 7.5e-7 / a    #0.56#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
Lambda_D = 0
h = 1e-3 #1e-2
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
g_xx = 1
g_xy = 0
g_yy = 1
g_yx = 0
n_cores = 8
points = 3*n_cores
params = {"L_x": L_x, "L_y": L_y, "w_0": w_0,
          "mu": mu, "Delta": Delta, "theta": theta,
           "Lambda_R": Lambda_R,
          "h": h , "k_x_values": k_x_values,
          "k_y_values": k_y_values, "h": h,
          "Lambda_D": Lambda_D,
          "g_xx": g_xx, "g_xy": g_xy, "g_yx": g_yx, "g_yy": g_yy,
          "points": points}


if __name__ == "__main__":
    # B_values = np.linspace(0, 6*Delta, points)
    B_values = np.append(np.linspace(0, 3*Delta, 2*points//3), np.linspace(3*Delta, 6*Delta, points//3))  #meV
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    n_B_y = np.array(results_pooled)
    
    data_folder = Path("Data/")
    
    name = f"n_By_mu_{mu}_L={L_x}_h={h}_B_y_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_Delta={Delta}_lambda_R={Lambda_R}_lambda_D={Lambda_D}_g_xx={g_xx}_g_xy={g_xy}_g_yy={g_yy}_g_yx={g_yx}_theta={np.round(theta,2)}_points={points}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , n_B_y=n_B_y, B_values=B_values,
             **params)
    print("\007")