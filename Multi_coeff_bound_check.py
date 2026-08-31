# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:32:01 2026

@author: se22pmat003
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Rmnj_scaling_mult_coeff_bound_check import Rmnj_scaling_mult_coeff_bound_check

# # Preallocate containers
# setup_time = np.zeros((10, 1))
# elapsed_time = [[None for _ in range(2)] for _ in range(10)]
# fmm_fun_val = [None for _ in range(10)]
# fmm_y = [None for _ in range(10)]
# flop_counts = [None for _ in range(10)]

# exact_fun_val = [None] * 10
# exact_y = [None] * 10

x_values = []

# fmm_failures = []
# exact_failures = []

# Parameters
scaling = 1
n_iter = 10  # Newton iterations
r = 20

n_values = np.arange(5000, 5001, 5000)

rng = np.random.default_rng(12345)

for jj, n in enumerate(n_values):
    
    print(f"jj = {jj}, n = {n}")

    # Generate 2n random points
    z = rng.uniform(0.0, 1.0, 2 * n)
    z.sort()

    # Interlaced x and y
    x = z[0::2]
    y = z[1::2]
    q = rng.uniform(0.5, 1.0, n)
    q = q**2  # elementwise square
    x_values.append(x.copy())
    
    results_1, results_2 = Rmnj_scaling_mult_coeff_bound_check(r, x, y, q, scaling)

    """
    results_1 captures only for the first leaf cell encountered and results_2 stores the 
    data of all the leaf cells.
    """    

df = pd.DataFrame(results_1)

print(df[[
    "m",
    "abs_ahat_m",
    "theoretical_bound",
    "ratio",
    "satisfies_bound"
]])


# Convert results_2 to DataFrame
df2 = pd.DataFrame(results_2)

# Leaf-cell index
leaf_cells = np.arange(1, len(df2) + 1)

# Numerical |ahat_m|
numerical = df2["abs_ahat_m"].to_numpy()

# Theoretical bound
theoretical = df2["theoretical_bound"].to_numpy()

# plt.figure(figsize=(9, 5))

# plt.scatter(
#     leaf_cells,
#     numerical,
#     s=25,
#     marker='o',
#     label=r'Numerical $|\widehat a_{10}|$'
# )

# plt.scatter(
#     leaf_cells,
#     theoretical,
#     s=25,
#     marker='x',
#     label=r'Analytical bound'
# )

# plt.xlabel('Leaf-cell index')
# plt.ylabel(r'Coefficient magnitude')
# plt.title(r'Validation of the Ramanujan-scaled bound for $m=10$')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

#------------------------ normal plot ---------------------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    leaf_cells,
    numerical,
    marker='o',
    markersize=4,
    label=r'Numerical $|\widehat a_{10}|$'
)

plt.plot(
    leaf_cells,
    theoretical,
    marker='s',
    markersize=4,
    linestyle='--',
    label=r'Analytical bound'
)

plt.xlabel('Leaf-cell index')
plt.ylabel(r'Coefficient magnitude')
plt.title(r'Validation of the Ramanujan-scaled bound for $m=10$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
