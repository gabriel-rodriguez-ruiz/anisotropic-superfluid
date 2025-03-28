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
file_to_open = data_folder / "n_By_mu_-39_L=1000_h=0.01_B_y_in_(0.0-0.3)_Delta=0.2_lambda_R=0.56_lambda_D=0.3_g_xx=1_g_xy=-2_g_yy=4_g_yx=2_theta=0_points=24.npz"
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
ax.plot(B_values/Delta, n_B_y[:,0], "-o",  label=r"$n_{s,xx}$")
ax.plot(B_values/Delta, n_B_y[:,1], "-o",  label=r"$n_{s,yy}$")
ax.plot(B_values/Delta, n_B_y[:,2], "-o",  label=r"$n_{s,xy}$")
ax.plot(B_values/Delta, n_B_y[:,3], "-o",  label=r"$n_{s,yx}$")
ax.plot(B_values/Delta, (n_B_y[:,0]+n_B_y[:,1]+n_B_y[:,2]+n_B_y[:,3])/2, "-o",  label=r"$n_{s,xx}+n_{s,yy}+n_{s,xy}+n_{s,yx}$")

ax.set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{np.round(theta,2)}"
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
plt.show()

#%%
data_folder = Path("Data/")
file_to_open = data_folder / "n_By_mu_-39_L=1000_h=0.01_B_y_in_(0.0-0.3)_Delta=0.2_lambda_R=0.56_lambda_D=0.3_g_xx=4_g_xy=-2_g_yy=1_g_yx=2_theta=0_points=24.npz"
Data = np.load(file_to_open)

file_to_open2 = data_folder / "n_By_mu_-39_L=1000_h=0.01_B_y_in_(0.0-0.3)_Delta=0.2_lambda_R=0.56_lambda_D=0.3_g_xx=4_g_xy=-2_g_yy=1_g_yx=2_theta=1.57_points=24.npz"
Data2 = np.load(file_to_open2)

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

n_B_y2 = Data2["n_B_y"]
B_values2 = Data2["B_values"]
Lambda_R2 = Data2["Lambda_R"]
Lambda_D2 = Data2["Lambda_D"]
Delta2 = Data2["Delta"]
theta2 = Data2["theta"]
w_02 = Data2["w_0"]
mu2 = Data2["mu"]
L_x2 = Data2["L_x"]
h2 = Data2["h"]
g_xx2 = Data2["g_xx"]
g_yy2 = Data2["g_yy"]
g_xy2 = Data2["g_xy"]
g_yx2 = Data2["g_yx"]

fig, ax = plt.subplots()
# ax.plot(B_values/Delta, n_B_y[:,0], "-o",  label=r"$n_{s,xx}$")
ax.plot(B_values/Delta, n_B_y[:,1], "-o",  label=r"$n_{s,yy}(\theta=0)$")
# ax.plot(B_values/Delta, n_B_y[:,2], "-o",  label=r"$n_{s,xy}$")
# ax.plot(B_values/Delta, n_B_y[:,3], "-o",  label=r"$n_{s,yx}$")
# ax.plot(B_values/Delta, (n_B_y[:,0]+n_B_y[:,1]+n_B_y[:,2]+n_B_y[:,3])/2, "-o",  label=r"$n_{s,xx}+n_{s,yy}+n_{s,xy}+n_{s,yx}$")

# ax.plot(B_values/Delta, n_B_y[:,0], "-o",  label=r"$n_{s,xx}$")
ax.plot(B_values2/Delta2, n_B_y2[:,1], "-o",  label=r"$n_{s,yy}(\theta=\pi/2)$")
# ax.plot(B_values/Delta, n_B_y[:,2], "-o",  label=r"$n_{s,xy}$")
# ax.plot(B_values/Delta, n_B_y[:,3], "-o",  label=r"$n_{s,yx}$")
# ax.plot(B_values/Delta, (n_B_y[:,0]+n_B_y[:,1]+n_B_y[:,2]+n_B_y[:,3])/2, "-o",  label=r"$n_{s,xx}+n_{s,yy}+n_{s,xy}+n_{s,yx}$")


ax.set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{np.round(theta,2)}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"
             +f"; h={h}" + "\n"
             + r"$\lambda_D=$" + f"{Lambda_D}"
             + r"; $g_{xx}=$" + f"{g_xx}"
             + r"; $g_{yy}=$" + f"{g_yy}"
             + r"; $g_{xy}=$" + f"{g_xy}"
             + r"$; g_{yx}=$" + f"{g_yx}")

ax.set_xlabel(r"$\frac{\mu_BgB}{\Delta}$")
ax.set_ylabel(r"$n_s$")
ax.legend()

# plt.tight_layout()
plt.show()
