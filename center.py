#!/usr/bin/python3

""" @package Center
  This module provide the algorithm for the IP approach.
"""

# Packages
import numpy as np
import time
from random import random
import time
import gurobipy as gp
import tqdm
from gurobipy import GRB

from utils import *

########################################################################################################
########################################################################################################


def I_0(template):
    """Return the index where all the vectors have the same value.
    @param template The database.
    @return The list of index where all vectors has the same value.
    """
    L = []
    boo = True
    n = len(template[0])
    m = len(template)
    for i in range(n):
        for j in range(m):
            if template[j][i] != template[(j+1) % m][i]:
                boo = False
                break
        if boo:
            L.append(i)
    return L

########################################################################################################
########################################################################################################


def Find_eps_cov_temp(template, tau):
    """ This function compute the master-template of a database.
    @param template The database.
    @param tau The threshold.
    @return The master-template of the database.
    """
    l = len(template)
    n = len(template[0])
    template = np.array(template)

    gp.setParam('OutputFlag', False)

    m = gp.Model("mat")

    p = m.addMVar(n, vtype=GRB.BINARY, name="p")

    I = I_0(template)
    for i in I:
        m.addConstr(p[i] == template[0][i])

    for i in range(l):
        y = [-2*template[i][j] for j in range(n)] + [sum(template[i])] + [1]
        m.addConstr(sum([p[i]*y[i] for i in range(n)]) +
                    y[n]+y[n+1]*sum(p) <= tau)
        m.addConstr(sum([p[i]*y[i] for i in range(n)])+y[n]+y[n+1]*sum(p) >= 0)

    #print("Compute center in R\n")
    v = compute_center_in_R(template)
    #print("Center in R : \n", v)

    #print("\nConstruct absolute values\n")
    gamma = m.addMVar(n, vtype=GRB.INTEGER, name="gamma")
    for i in range(n):
        m.addConstr(-gamma[i] <= p[i] - v[i])
        m.addConstr(gamma[i] >= p[i] - v[i])
        m.addConstr(gamma[i] >= 0)

    #print("Set objective\n")
    m.setObjective(sum(gamma), GRB.MINIMIZE)

    m._vars = m.getVars()
    # print("Start solving\n")
    m.Params.MIPGapAbs = 0.
    m.Params.MIPGap = 0
    m.Params.NumericFocus = 3
    m.Params.IntFeasTol = 10**(-9)
    m.Params.FeasibilityTol = 10**(-9)
    m.Params.OptimalityTol = 10**(-9)

    start_timer()
    m.optimize()
    stop_timer()

    nSolutions = m.SolCount

    verif = None
    if nSolutions >= 1:
        for i in range(nSolutions):
            m.params.SolutionNumber = i
            result = m.getAttr(GRB.Attr.X, m.getVarByName("p"))
            res = [int(result[i]) for i in range(n)]
            verif = verification(res, template, tau)
            print(verif)
            if verif:
                break
    else:
        res = None
        verif = False
    return res, verif, get_time()


# size_of_template = 10
# nbr_client = 15
# tau = 3
# temp = gen_near_templates(size_of_template, nbr_client, tau, 12)
#
# print("Template = ", temp)
#
# Find_eps_cov_temp(temp, tau)


Res_thread = []


def several_iter2(n, epsilon, nbr_cli, sed, repet):
    """Set up for experiments"""
    temps = 0
    global Res_thread
    for _ in tqdm.tqdm(range(repet)):
        templatelist = gen_near_templates(
            n, nbr_cli, epsilon, sed=sed*random())
        tmps1 = time.perf_counter()
        res, verif, tmps2 = Find_eps_cov_temp(templatelist, epsilon)
        temps += (tmps2-tmps1)
    Res_thread.append(temps)
    return True


def main_test_mt(L, rep=10000):
    """Launch tests to find an epsilon master template.
    @param L The vector of parameters
    """
    global Res_thread
    tmpslst = []
    Gres = []
    tmp = random()
    for liste in L:
        tmp *= random()
        Res_thread = []
        n = liste[0]
        epsilon = liste[1]
        nbr_cli = liste[2]
        several_iter2(n, epsilon, nbr_cli, random()*tmp, rep)
        k = sum(Res_thread)

        Gres.append([n, epsilon, nbr_cli,
                     round((k/rep)*(10**3), 3)])
        print(Gres)

    print("$n$ & $\\epsilon $ & $\\#$clients & Time")
    for liste in Gres:
        print("$", liste[0], "$ & $", liste[1], "$ & $", liste[2], "$ & $",
              liste[3], "$\\\\\n")
    return tmpslst


# L = []
# for i in range(15, 50, 5):
#     L.append([i, 10, 50])
# L.append([70, 3, 200])
# for i in range(5, 40, 5):
#     L.append([70, i, 200])
# for i in range(30, 200, 20):
#     L.append([70, 10, i])
# 
# main_test_mt(L, rep=100)
