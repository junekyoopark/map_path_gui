from gurobipy import *
import numpy as np
import copy
import time
from matplotlib import pyplot as plt

def optimize_speed( kapparef, sigmaref, arclength, is_closed ):
# OPTIMIZE_SPEED Calculate longitudinal velocity, acceleration and jerk based on point locations
#
#    Input(s):
#    (1) kappa_vec - Curvature of points, where each line is a point;
#    (2) s_vec     - Arclength of points w.r.t start, where each line is a point;
#    (1) is_closed - Boolean flag indicating if path is closed loop.
#
#    Output(s):
#    (1) Profile - Structure containing the data defining the optimal speed profile.
#            Profile.t - Timestamps for points;
#            Profile.velocity_x - Longitudinal velocity [m/s];
#            Profile.accel_x - Longitudinal acceleration [m/s^2];
#            Profile.jerk_x - Longitudinal jerk [m/s^3].
#
#    Author(s):
#    (1) Júnior A. R. Da Silva
#    (2) Carlos M. Massera
#
#    Copyright owned by Universidade de São Paulo

    s_vec = arclength

    v_ref = 22
    jerkLatMax = 20.#1.2
    accLatMax = 2.0
    kMaximumAx = 5.0
    kMaximumJx = 5.0
    kMaxIter = 50;       # Maximum number of iterations
    kTolNormInf = 1e-2;  # Tolerance for stop criteria

    # velocity_x_max = v_ref*np.ones(len(kapparef))
    velocity_x_max = np.zeros(len(kapparef))
    for k in range(0, len(kapparef)):
        velocity_x_max[k] = min(min((jerkLatMax/abs(sigmaref[k]))**(1./3.),\
            v_ref),np.sqrt(accLatMax/abs(kapparef[k])))
    plt.plot(arclength, velocity_x_max)
    # Data vector size
    n_x = len(kapparef)

    # Arclength variation
    d_s = np.diff(s_vec)

    # Set previous speed to zero
    v_x_prev = np.zeros(len(velocity_x_max))
    v_x_out = copy.copy(velocity_x_max)

    # Initialize jounce weights to one
    # Jounce weighting (for piecewise jerk)
    jounce_w = np.ones(n_x - 3)       # Jounce weighting for zero norm approximation

    # Sequantial LP solver outter loop
    exit_success = False

    for i in range(kMaxIter):

        timer = time.time()

        # Create a new model
        m = Model("lp")

        # Profile variables
        velocity_x = m.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY)       # Longitudinal velocity [m/s]
        # velocity_x_prev = m.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY)  # Previous longitudinal velocity [m/s]
        accel_x = m.addVars(n_x-1, lb=-GRB.INFINITY, ub=GRB.INFINITY)        # Longitudinal acceleration [m/s^2]
        jerk_x = m.addVars(n_x-2, lb=-GRB.INFINITY, ub=GRB.INFINITY)         # Longitudinal jerk [m/s^3]
        jounce_x = m.addVars(n_x-3, lb=-GRB.INFINITY, ub=GRB.INFINITY)       # Longitudinal jounce [m/s^4]
        d_v = m.addVars(n_x-1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        d_a = m.addVars(n_x-2, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        d_j = m.addVars(n_x-3, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        abs_jounce_x = m.addVars(n_x-3, lb=-GRB.INFINITY, ub=GRB.INFINITY)

        # Define initial guess as maximum available speed
        velocity_x_prev = copy.copy(v_x_out)

        # Dynamics
        # Since a = dv/dt = (dv/ds)(ds/dt) = (dv/ds) v
        # We've got dv = a ds / v
        for p in range(n_x-1):
            d_v[p] = d_s[p] * (accel_x[p] / velocity_x_prev[p])

        # And similarly for acceleration and jerk
        for p in range(n_x-2):
            d_a[p] = d_s[p] * (jerk_x[p] / velocity_x_prev[p])
        for p in range(n_x-3):
            d_j[p] = d_s[p] * (jounce_x[p] / velocity_x_prev[p])


        # Setup optimization problem
        # Minimize -||velocity_x||_1 + ||jounce_x||_0
        m.addConstrs( ( 0 <= velocity_x[p] ) for p in range(n_x) )
        m.addConstrs( ( velocity_x[p] <= velocity_x_max[p] ) for p in range(n_x) )
        m.addConstrs( ( -kMaximumAx <= accel_x[p] ) for p in range(n_x-1) )
        m.addConstrs( ( accel_x[p] <= kMaximumAx ) for p in range(n_x-1) )
        m.addConstrs( ( -kMaximumJx <= jerk_x[p] ) for p in range(n_x-2) )
        m.addConstrs( ( jerk_x[p] <= kMaximumJx ) for p in range(n_x-2) )
        m.addConstrs( ( velocity_x[p+1] - velocity_x[p] == d_v[p] ) for p in range(n_x-1) )
        m.addConstrs( ( accel_x[p+1] - accel_x[p] == d_a[p] ) for p in range(n_x-2) )
        m.addConstrs( ( jerk_x[p+1] - jerk_x[p] == d_j[p] ) for p in range(n_x-3) )
        for p in range(n_x-3):
            m.addGenConstrAbs( abs_jounce_x[p],  jounce_x[p] )

        # Closed loop constraints
        if is_closed:
            # If trajectory is closed loop, keep speed and accel continuous
            m.addConstr( velocity_x[0] == velocity_x[n_x-1] )
            m.addConstr( accel_x[0] == accel_x[n_x-2] )
        else:
            m.addConstr( velocity_x[0] == 10.0 )
            m.addConstr( accel_x[0] == 0.5 )
            m.addConstr( jerk_x[0] == 0.5 )
        #     m.addConstr( velocity_x[n_x-1] == 0 )
        #     m.addConstr( accel_x[n_x-2] == 0 )

        # We want maximum speed and piecewise linear acceleration, therefore
        objective  = 0
        for p in range(n_x):
            if p < n_x-3:
                objective = objective - velocity_x[p] + (abs_jounce_x[p]*abs(jounce_w[p]))        # And minimize approximation of jounce zero norm
            else:
                objective = objective - velocity_x[p]
        m.setObjective(objective, GRB.MINIMIZE)

        # Call the solver
        # Take half step for better convergence since we don't have step constraints
        m.Params.OutputFlag = 0
        m.optimize()
        v_x_prev = copy.copy(v_x_out)

        for p in range(n_x):
            # print(velocity_x[p].x)
            v_x_out[p] = (velocity_x[p].x + v_x_out[p]) / 2

        # Calculate current values
        t_vec = np.concatenate((np.zeros(1), np.cumsum(d_s / v_x_out[0:len(v_x_out)-1])), axis=0)
        accel_x = np.diff(v_x_out) / np.diff(t_vec)
        jerk_x = np.diff(accel_x) / np.diff(t_vec[0:len(t_vec)-1])
        jounce_x = np.diff(jerk_x) / np.diff(t_vec[0:len(t_vec)-2])

        # Update jounce weights
        jounce_w = 1./(abs(jounce_x) + 1e-5)

        error = np.linalg.norm(v_x_prev - v_x_out, np.inf)
        print('    Iteration ' + str(i+1) + ' - Error: ' + str(error) + ' Elapsed: ' + \
                str((time.time()-timer) * 100000.0 / 100.0) + ' miliseconds.')

        # Stop if error met
        if error <= kTolNormInf:
            exit_success = True
            print('    Completed - Error tolerance met.')
            break

    if exit_success == False:
        print('    Completed - Exceeded number of iterations.')
    return v_x_out

waypoints = np.load('waypoints.npy')
kapparef = waypoints[:,2]
sigmaref = waypoints[:,6]
arclength = waypoints[:,4]
is_closed = False
v_x_out = optimize_speed( kapparef, sigmaref, arclength, is_closed )
waypoints[:,5] = v_x_out
print(sum(v_x_out))
#Clot 26983.33553113216
#Center 26329.04524715995
#Manual 25959.337341408565


file = 'waypoints.npy'
np.save(file, waypoints)

plt.plot(arclength, v_x_out)
plt.xlabel("Distance [m]")
plt.ylabel("Speed [m/s]")
plt.show()
