# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:32:01 2026

@author: se22pmat003
Check the following inequality

\begin{equation}
\left\|\widehat B^{(R)}\right\|_F
\leq
\frac{1}{|d|}
\left[
1+
\frac{4\pi e^{1/6}\eta^2}{(1-\eta^2)^2}
+
\frac{4\pi^2e^{1/3}}{6}
\sum_{k=2}^{r-1}
k(k^2-1)q^{2k}
\right]^{1/2}.
\end{equation}

for all M2L interactions of the FMM tree

"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

from Rmnj_scaling_Frob_norm import Rmnj_scaling_Frob_norm


x_values = []

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
    
    
    M2L_results = Rmnj_scaling_Frob_norm(r, x, y, q, scaling)

    df_M2L = pd.DataFrame(M2L_results)

    print(df_M2L[
        [
            "target_node",
            "source_node",
            "level",
            "d",
            "q",
            "eta",
            "numerical_frobenius",
            "theoretical_bound",
            "ratio",
            "satisfies_bound"
        ]
    ])

    #----------------------- Frobenius norm plot --------------------------------------
    
    # plt.figure(figsize=(9, 5))
    
    # plt.plot(
    #     np.arange(len(df_M2L)),
    #     df_M2L["numerical_frobenius"],
    #     marker='o',
    #     markersize=3,
    #     label=r'Numerical $\|\widehat B^{(R)}\|_F$'
    # )
    
    # plt.plot(
    #     np.arange(len(df_M2L)),
    #     df_M2L["theoretical_bound"],
    #     linestyle='--',
    #     label=r'Analytical upper bound'
    # )
    
    # plt.xlabel("M2L interaction")
    # plt.ylabel(r"Frobenius norm")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    # #---------------------- ratio plot ------------------------
    
    # plt.figure(figsize=(9, 5))

    # plt.plot(
    #     np.arange(len(df_M2L)),
    #     df_M2L["ratio"],
    #     marker='o',
    #     markersize=3
    # )
    
    # plt.axhline(
    #     1.0,
    #     linestyle='--',
    #     linewidth=1.2,
    #     label=r'Upper-bound threshold ($R=1$)'
    #         )
    
    # plt.xlabel("M2L interaction")
    # plt.ylabel(
    #     r"$\|\widehat B^{(R)}\|_F/"
    #     r"\|\widehat B^{(R)}\|_{F,\mathrm{bound}}$"
    # )
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    
    # print("Maximum ratio:", df_M2L["ratio"].max())
    # print("All bounds satisfied:",
    #       df_M2L["satisfies_bound"].all())


#----------------- scatter plot Frobenius norm -------------------

    plt.figure(figsize=(9, 5))

    plt.scatter(
        np.arange(len(df_M2L)),
        df_M2L["numerical_frobenius"],
        s=25,
        label=r'Numerical $\|\widehat B^{(R)}\|_F$'
    )
    
    plt.scatter(
        np.arange(len(df_M2L)),
        df_M2L["theoretical_bound"],
        s=25,
        marker='x',
        label=r'Analytical upper bound'
    )
    
    plt.xlabel("M2L interaction indices")
    plt.ylabel(r"Frobenius norm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
  #---------------------------ratio scatter plot -------------------
    plt.figure(figsize=(9, 5))

    plt.scatter(
        np.arange(len(df_M2L)),
        df_M2L["ratio"],
        s=25,
        marker='o',
        label=r'Ratio'
    )
    
    plt.axhline(
        1.0,
        linestyle='--',
        linewidth=1.2,
        label=r'Upper-bound threshold ($R=1$)'
            )
    
    plt.xlabel("M2L interaction indices")
    plt.ylabel(
        r"$\|\widehat B^{(R)}\|_F/"
        r"\|\widehat B^{(R)}\|_{F,\mathrm{bound}}$"
    )
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Maximum ratio:", df_M2L["ratio"].max())
    print("All bounds satisfied:",
          df_M2L["satisfies_bound"].all())  