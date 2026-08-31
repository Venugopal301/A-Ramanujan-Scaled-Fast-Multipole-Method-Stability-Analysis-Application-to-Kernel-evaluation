#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:13:39 2026

@author: venugopal
    Validate the bound

        |ahat_m| < Q_s * alpha_s^m / sqrt(2*pi*m),

    for m = 1,...,r-1 and for leaf cell first encountered in the FMM post-order traversal.

"""
import math
import numpy as np


def validate_ramanujan_multipole_bound( x, q, I, Px, ch, post_ord, r=20 ):
    
    
    
        # ---------------------------------------------------------
        # Ramanujan factor Gamma_r
        # ---------------------------------------------------------
    Gamma_r = (
        np.sqrt(np.pi)
        * (
            8 * r**3
            + 4 * r**2
            + r
            + 1 / 30
        )**(1 / 6)
    )**(1 / r)

    results = []

    # =========================================================
    # Post-order traversal
    # =========================================================
    for i in post_ord:

        idx = i - 1
        lvl = int(I[idx, 6])

        # -----------------------------------------------------
        # Consider only leaf source cells
        # -----------------------------------------------------
        if len(ch[idx]) != 0:
            continue

        # Optional: restrict to the same levels used in P2L/P2M
        if lvl < 2:
            continue

        # -----------------------------------------------------
        # Source cell information
        # -----------------------------------------------------
        c_s = I[idx, 0]
        delta_s = I[idx, 1]

        source_idx = Px[idx]

        # Skip an empty leaf
        if len(source_idx) == 0:
            continue

        x_s = x[source_idx]
        q_s = q[source_idx]

        # Number of particles in this source cell
        n_s = len(source_idx)

        # -----------------------------------------------------
        # Q_s
        # -----------------------------------------------------
        Q_s = np.sum(np.abs(q_s))

        # -----------------------------------------------------
        # rho_s = max_j |x_j - c_s|
        # -----------------------------------------------------
        rho_s = np.max(np.abs(x_s - c_s))
         
        # -----------------------------------------------------
        # alpha_s
        # -----------------------------------------------------
        alpha_s = rho_s * Gamma_r / delta_s

        # =====================================================
        # Expansion orders m = 1,...,r-1
        # =====================================================
        for m in range(1, r):

            # -------------------------------------------------
            # a_m
            #
            # a_m = sum_j q_j (x_j-c_s)^m / m!
            # -------------------------------------------------
            a_m = (
                np.sum(
                    q_s * (x_s - c_s)**m
                )
                / math.factorial(m)
            )

            # -------------------------------------------------
            # Ramanujan scaling theta_{s,m}
            # -------------------------------------------------
            theta_sm = (
                (m / np.e)
                * Gamma_r
                / delta_s
            )**m

            # -------------------------------------------------
            # Scaled multipole coefficient
            # -------------------------------------------------
            ahat_m = a_m * theta_sm

            numerical_value = np.abs(ahat_m)

            # -------------------------------------------------
            # Theoretical bound
            #
            # Q_s alpha_s^m / sqrt(2*pi*m)
            # -------------------------------------------------
            # theoretical_value = (
            #     Q_s
            #     * alpha_s**m
            #     / np.sqrt(2 * np.pi * m)
            # )
            theoretical_value = (
                Q_s
                * alpha_s**m
                / np.sqrt(2 * np.pi * m)
                * np.exp(-1.0 / (12 * m + 1)))

            # -------------------------------------------------
            # Ratio
            # -------------------------------------------------
            if theoretical_value != 0:
                ratio = (
                    numerical_value
                    / theoretical_value
                )
            else:
                ratio = np.nan

            results.append({
                "node": i,
                "level": lvl,
                "n": n_s,
                "m": m,
                "c_s": c_s,
                "delta_s": delta_s,
                "rho_s": rho_s,
                "Q_s": Q_s,
                "Gamma_r": Gamma_r,
                "alpha_s": alpha_s,
                "a_m": a_m,
                "theta_sm": theta_sm,
                "abs_ahat_m": numerical_value,
                "theoretical_bound": theoretical_value,
                "ratio": ratio,
                "satisfies_bound":
                    numerical_value < theoretical_value
            })
        break    
    return results    
