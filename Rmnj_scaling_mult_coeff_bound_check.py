# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:09:37 2026

@author: se22pmat003

compare the bound
$|\widehat a_m|$ \leq 
\[
\frac{Q_s\alpha_s^m}{\sqrt{2\pi m}}
\exp\left(-\frac{1}{12m+1}\right).
\]

for m = 1:r for first leaf cell encountered in the post-order traversal and for m = 10
for all leaf cells of the FMM tree
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

import numpy as np
from child1 import child1
from postorder import postorder
from preorder import preorder
from sib import sib
from validate_ramanujan_multipole_bound import validate_ramanujan_multipole_bound
from validate_ramanujan_multipole_bound_2 import validate_ramanujan_multipole_bound_2


def Rmnj_scaling_mult_coeff_bound_check(r, x, y, q, scaling):

    N0 = 64
    tau = 0.6

    z2 = max(np.max(x), np.max(y))
    z1 = min(np.min(x), np.min(y))

    z2 += 0.1 * abs(z2)
    z1 -= 0.1 * abs(z1)

    # Root box
    I = np.array([[(z1 + z2) / 2, (z2 - z1) / 2, 1, 1, 1, 0, 1]], dtype=float)

    S = []

    S.append(I)

    Px = [np.arange(len(x))]
    Py = [np.arange(len(y))]

    while len(S) > 0:

        i = S.pop().ravel()

        c = i[0]
        d = i[1]
        el = int(i[2])
        er = int(i[3])
        idx = int(i[4])
        lvl = int(i[6])

        n = I.shape[0]

        px = Px[idx - 1]
        Px[idx - 1] = np.array([], dtype=int)

        py = Py[idx - 1]
        Py[idx - 1] = np.array([], dtype=int)

        # -----------------------------------
        # Bisect interval
        # -----------------------------------
        i1 = np.array([[c - d / 2, d / 2, el, 1, n + 1, idx, lvl + 1]])
        i2 = np.array([[c + d / 2, d / 2, 0, er, n + 2, idx, lvl + 1]])

        I = np.vstack((I, i1, i2))

        # -----------------------------------
        # Left child
        # -----------------------------------

        if el == 1:

            Px.append(px[(c - d <= x[px]) & (x[px] <= c)])

            Py.append(py[(c - d <= y[py]) & (y[py] <= c)])

        else:

            Px.append(px[(c - d < x[px]) & (x[px] <= c)])

            Py.append(py[(c - d < y[py]) & (y[py] <= c)])

        # -----------------------------------
        # Right child
        # -----------------------------------

        if er == 1:

            Px.append(px[(c < x[px]) & (x[px] <= c + d)])

            Py.append(py[(c < y[py]) & (y[py] <= c + d)])

        else:

            Px.append(px[(c < x[px]) & (x[px] < c + d)])

            Py.append(py[(c < y[py]) & (y[py] < c + d)])

        # -----------------------------------
        # Adaptive subdivision
        # -----------------------------------
        left_idx = len(Px) - 2
        right_idx = len(Px) - 1

        if len(Px[left_idx]) > N0 or len(Py[left_idx]) > N0:
            S.append(i1)

        if len(Px[right_idx]) > N0 or len(Py[right_idx]) > N0:
            S.append(i2)

    # Parent array
    tr = I[:, 5].astype(int)

    # Child array
    ch = child1(tr)

    # Number of nodes
    n = len(tr)

    # Traversals
    post_ord = postorder(1, tr, ch)
    pre_ord = preorder(1, tr, ch)

    # -----------------------------------
    # Neighbor and interaction lists
    # -----------------------------------

    neighbor = [[] for _ in range(n)]
    interlist = [[] for _ in range(n)]

    for i in pre_ord:
        parent = int(tr[i - 1])
        if parent == 0:
            neighbor[i - 1] = []
            interlist[i - 1] = []
        else:
            neighbor[i - 1] = [sib(tr, ch, i)]
            cousins = []
            for j in neighbor[parent - 1]:
                if len(ch[j - 1]) == 0:
                    cousins.append(j)
                else:
                    cousins.append(ch[j - 1][0])
                    cousins.append(ch[j - 1][1])
            ci = I[i - 1, 0]
            ri = I[i - 1, 1]
            for j in cousins:
                cj = I[j - 1, 0]
                rj = I[j - 1, 1]
                if ri + rj <= tau * abs(ci - cj):
                    interlist[i - 1].append(j)
                else:
                    neighbor[i - 1].append(j)

    for i in pre_ord:
        if len(ch[i - 1]) == 0:
            for j in neighbor[i - 1]:
                tmp = set(neighbor[j - 1])
                tmp.add(i)
                neighbor[j - 1] = list(tmp)

            for j in interlist[i - 1]:
                tmp = set(interlist[j - 1])
                tmp.add(i)
                interlist[j - 1] = list(tmp)

   
    results_1 = validate_ramanujan_multipole_bound( x, q, I, Px, ch, post_ord, r=20 )
    results_2 = validate_ramanujan_multipole_bound_2( x, q, I, Px, ch, post_ord, r=20)
    
    """
    results_1 captures only for the first leaf cell encountered and results_2 stores the 
    data of all the leaf cells.
    """
    
    return results_1, results_2            




