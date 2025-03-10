#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:05:28 2024

@author: gabriel
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

data_folder = Path("Data/")
file_to_open = data_folder / "n_By_mu_-39_L=1000_h=0.01_B_y_in_(0.0-0.4)_Delta=0.2_lambda_R=0.6_lambda_D=0.75_g_xx=1_g_xy=0_g_yy=1_points=24.npz"
Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
Lambda_R = Data["Lambda_R"]
Lambda_D = Data["Lambda_D"]
Delta = Data["Delta"]
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
L_x = Data["L_x"]
h = Data["h"]
g_xx = Data["g_xx"]
g_yy = Data["g_yy"]
g_xy = Data["g_xy"]
g_yx = Data["g_yx"]

fig, ax = plt.subplots()
ax.plot(B_values/Delta, n_B_y[:,0], "-o",  label=r"$n_{s,\perp}$")
ax.plot(B_values/Delta, n_B_y[:,1], "-o",  label=r"$n_{s,\parallel}$")
ax.set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"
             +f"; h={h}" + "\n"
             + r"$\lambda_D=$" + f"{Lambda_D}"
             + r"; $g_{xx}=$" + f"{g_xx}"
             + r"; $g_{yy}=$" + f"{g_yy}"
             + r"; $g_{xy}=$" + f"{g_xy}"
             + r"$; g_{yx}=$" + f"{g_yx}")

ax.set_xlabel(r"$\frac{B}{\Delta}$")
ax.set_ylabel(r"$n_s$")
ax.legend()

# plt.tight_layout()
# plt.show()

#%%
data_folder = Path("Data/")
file_to_open = data_folder / "n_By_mu_-40_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda=0.npz"
Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
Lambda = Data["Lambda"]
Delta = Data["Delta"]
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
L_x = Data["L_x"]
h = Data["h"]

ax.plot(B_values/Delta, n_B_y[:,0], "-o",  label=r"$n_{s,\perp}(\lambda=$"+f"{Lambda})")
ax.plot(B_values/Delta, n_B_y[:,1], "-o",  label=r"$n_{s,\parallel}(\lambda=$"+f"{Lambda})")
ax.legend()

plt.tight_layout()
plt.show()