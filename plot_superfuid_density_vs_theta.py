#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:23:07 2024

@author: gabriel
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

data_folder = Path("Data/")
file_to_open = data_folder / "n_theta_mu_-39_L=1000_h=0.01_theta_in_(0.0-3.142)B=0.18_Delta=0.2_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=1_g_yy=1_points=24.npz"
Data = np.load(file_to_open, allow_pickle=True)

n_theta = Data["n_theta"]
theta_values = Data["theta_values"]
Lambda_R = Data["Lambda_R"]
Lambda_D = Data["Lambda_D"]
Delta = float(Data["Delta"])
w_0 = Data["w_0"]
mu = Data["mu"]
L_x = Data["L_x"]
B = Data["B"]
g_xx = Data["g_xx"]
g_yy = Data["g_yy"]
g_xy = Data["g_xy"]
g_yx = Data["g_yx"]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

ax.plot(theta_values, n_theta[:,0], "-o",  label=r"$n_{s,xx}$")
ax.plot(theta_values, n_theta[:,1], "-o",  label=r"$n_{s,yy}$")
ax.set_title(r"$\lambda_R=$" + f"{Lambda_R:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $B=$" + f"{B:.3}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"+ "\n"
             +r"$\lambda_D=$" + f"{np.round(Lambda_D,2)}"
             + r"; $g_{xx}=$" + f"{g_xx}"
             + r"; $g_{yy}=$" + f"{g_yy}"
             + r"; $g_{xy}=$" + f"{g_xy}"
             + r"; $g_{yx}=$" + f"{g_yx}")
# ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$n_s(B_y)$")
ax.legend()
plt.tight_layout()
plt.show()