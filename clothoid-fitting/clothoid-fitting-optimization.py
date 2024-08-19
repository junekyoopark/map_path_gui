import numpy as np
from matplotlib import pyplot as plt
import copy
from scipy import signal
import math
from scipy.interpolate import interp1d
import time
from gurobipy import *

from numpy import genfromtxt
import os

output_dir = './temp/'

def pre_solve( path, kappa, arclength, kappa_max ):

# PRE_SOLVE Generates good initialization for curvature optimization
#
#    Input(s):
#    (1) path      - X and Y position of points, where each line is a (X, Y) pair;
#    (2) kappa     - Column vector with curvatures for each point;
#    (3) arclength - Column vector with arclength from first point to each point.
#
#    Output(s):
#    (1) PreSolveResults - Structure containing the presolve solutions.
#            PreSolveResults.magnitude  - Distance between adjacent points;
#            PreSolveResults.rot_matrix - Rotation matrix from world frame to path frame;
#            PreSolveResults.kappa      - Curvature initialization for optimize_curvature function.
#
#    Author(s):
#    (1) Júnior A. R. Da Silva
#    (2) Carlos M. Massera
#
#    Copyright owned by Universidade de São Paulo

    print('Pre-solver')
    timer = time.time()

    # Create a new model
    m = Model("lp")

    # Convergence constants
    kMaxDeltaKappa = 0.02  # Maximum curvature variation per iteration [1/m]
    kTolNormInf = 0.00001     # Error tolerance for stop criteria [m] DEFAULT 0.0001
    kMaxIter = 50000000          # Maximum number of iterations

    # Get data size
    data_size = len(path) - 1

    # Initialize output structure
    # PreSolveResults = struct();

    # Calculate integration matrix
    magnitude = np.diff(arclength)
    # PreSolveResults.magnitude = magnitude;  # Save magnitude since this is used later again

    # Reconstruct points from curvature data for reference
    # Note: Using cumsum instead of matrix S saves a lot of CPU cycles
    theta = np.cumsum(magnitude * kappa);  # Equivalent to: theta = S * kappa;
    x = np.cumsum(magnitude * np.cos(theta));  # Equivalent to: S * cos(theta);
    y = np.cumsum(magnitude * np.sin(theta));  # Equivalent to: S * sin(theta);
    # print(x)
    # Find rotation that recover inital points
    # Note: since each point is a row, rotation matrix mut be transposed and multiplied on
    #       the left hand side.
    r_ang = math.atan2(np.diff(path[0:2,1]), np.diff(path[0:2,0])) - math.atan2(y[0], x[0])
    cr = np.cos(r_ang)
    sr = np.sin(r_ang)
    rot_matrix = np.array([[cr, -sr], [sr, cr]])

    # print(np.array([x, y]) * rot_matrix.transpose())
    # PreSolveResults.rot_matrix = rot_matrix;  # Save rot_matrix since this is used later again
    error = np.linalg.norm( (rot_matrix.dot(np.array([x, y]))).transpose() - path[1:len(path), :], np.inf)

    # If error is sufficently small just exit
    if error < kTolNormInf:
        # PreSolveResults.kappa = kappa  # Return initial curvature values
        print('    Early completion - Error tolerance met.')
        return rot_matrix, kappa, magnitude

    kappa_out = kappa

    # Execute Sequential Linear Programming
    for i in range(0,kMaxIter):
        timer = time.time()

        # Define parameters
        kappa_lin = kappa_out
        theta_sk = np.cumsum(magnitude * kappa_lin)
        cos_sk = np.cos(theta_sk)
        sin_sk = np.sin(theta_sk)

        # Create a new model
        m = Model("lp")

        # Construct the optimization problem apriori to improve performance
        # Variables
        kappa_v = m.addVars(data_size, lb=-GRB.INFINITY, ub=GRB.INFINITY)       # Optimization variable
        theta_k = m.addVars(data_size, lb=-GRB.INFINITY, ub=GRB.INFINITY)     # (S * (kappa - kappa_lin))
        x_k = m.addVars(data_size+1, lb=-GRB.INFINITY, ub=GRB.INFINITY)        # (S * cos(theta))
        y_k = m.addVars(data_size+1, lb=-GRB.INFINITY, ub=GRB.INFINITY)       # (S * sin(theta))
        norm_slack = m.addVars(data_size+1, lb=-GRB.INFINITY, ub=GRB.INFINITY)  # Slack for norm approximation
        path_k = m.addVars(data_size+1,2, lb=-GRB.INFINITY, ub=GRB.INFINITY)

        # Minimize norm 1 path recovery error error
        obj =  norm_slack.sum('*')
        # obj = kappa_v.sum('*')
        
        # obj = 0
        # for p in range(data_size+1):
        #     obj = obj + norm_slack[p]

        m.setObjective(obj, GRB.MINIMIZE)

        for p in range(data_size+1):
            path_k[p,0] = rot_matrix[0,:].dot(([x_k[p],y_k[p]]))# +  rot_matrix[0,1]*y_k[p]
            path_k[p,1] = rot_matrix[1,:].dot(([x_k[p],y_k[p]]))

        # Constraint kappa variations to avoid leaving the linearization point and ensure the defnition
        #     of theta_k holds
        # Note that theta_k = cumsum(maginitude .* (kappa_v - kappa_lin)
        #     using the identity diff(cumsum(x)) = x(2:end)
        #     we can constraint diff(theta_k) = maginitude .* (kappa_v - kappa_lin) for 2:end
        #     and separatedely constraint theta_k(1).
        #     This ensures higher sparsity of the KKT matrix and improve execution times for LP solver
        
        m.addConstrs( ( -norm_slack[p] <= path_k[p,0] - path[p,0] ) for p in range(data_size+1) )
        m.addConstrs( ( -norm_slack[p] <= path_k[p,1] - path[p,1] ) for p in range(data_size+1) )
        m.addConstrs( ( path_k[p,0] - path[p,0] <= norm_slack[p] ) for p in range(data_size+1) )
        m.addConstrs( ( path_k[p,1] - path[p,1] <= norm_slack[p] ) for p in range(data_size+1) )
        m.addConstr( theta_k[0] == magnitude[0] * (kappa_v[0] - kappa_lin[0]) )
        m.addConstrs( theta_k[p]-theta_k[p-1] == magnitude[p] * (kappa_v[p] - kappa_lin[p]) for p in range(1,data_size) )
        m.addConstrs( x_k[p+1] - x_k[p] == magnitude[p] * (cos_sk[p] - sin_sk[p] * theta_k[p]) for p in range(0, data_size) )
        m.addConstrs( y_k[p+1] - y_k[p] == magnitude[p] * (sin_sk[p] + cos_sk[p] * theta_k[p]) for p in range(0, data_size) )
        m.addConstrs( -kMaxDeltaKappa <= kappa_v[p] - kappa_lin[p] for p in range(0, data_size) )
        m.addConstrs( kappa_v[p] - kappa_lin[p] <= kMaxDeltaKappa for p in range(0, data_size) )

        m.Params.OutputFlag = 0

        # Call the solver
        m.optimize()

        for p in range(data_size):
            kappa_out[p] = kappa_v[p].x

        # Calculate new error
        theta = np.cumsum(magnitude * kappa_out)  # Equivalent to: theta = S * kappa;
        x = np.cumsum(magnitude * np.cos(theta))  # Equivalent to: S * cos(theta);
        y = np.cumsum(magnitude * np.sin(theta))  # Equivalent to: S * sin(theta);

        error = np.linalg.norm( (rot_matrix.dot(np.array([x, y]))).transpose() - path[1:len(path), :], np.inf)
        print('    Iteration ' + str(i+1) + ' - Error: ' + str(error) + ' Elapsed: ' + \
              str((time.time()-timer) * 100000.0 / 100.0) + ' miliseconds.')


        if error < kTolNormInf:
            kappa = kappa_out  # Output optimized kappa
            print('    Completed - Error tolerance met.')
            return rot_matrix, kappa, magnitude
    #end for
    kappa = kappa_out  # Output optimized kappa
    print('    Completed - Exceeded number of iterations.')
    return rot_matrix, kappa, magnitude

def optimize_curvature( path, max_deviation, kappa_max ):

# OPTIMIZE_CURVATURE Generates a locally optimal trajectory with bounded deviation from points
#
#    Input(s):
#    (1) path          - X and Y position of points, where each line is a (X, Y) pair;
#    (2) max_deviation - Maximum inifnity norm error between trajectory and original points.
#
#    Output(s):
#    (1) Trajectory - Structure containing the data defining the optimal trajectory.
#            Trajectory.position  - Matrix where each line is the position of a point;
#            Trajectory.theta     - Matrix where each line is the heading of a point;
#            Trajectory.arclength - Matrix where each line is the arclength of a point;
#            Trajectory.kappa     - Matrix where each line is the curvature of a point;
#            Trajectory.dkappa    - Matrix where each line is the sharpness of a point.
#
#    Author(s):
#    (1) Júnior A. R. Da Silva
#    (2) Carlos M. Massera
#
#    Copyright owned by Universidade de São Paulo

    # Set constants for optimization
    kMaxDeltaKappa = 0.02;  # Maximum curvature variation per iteration [1/m]
    kTolNormInf = 5e-10;     # Error tolerance for stop criteria [1/m] Default 5e-4
    kMaxIter = 50000000;          # Maximum number of iterations

    # Get data size
    # Note that size is one smaller than data (you only need n arcs to connect n+1 points)
    data_size = len(path) - 1

    # Check if path is closed
    is_closed = (np.linalg.norm(path[0,:] - path[len(path)-1,:], np.inf) < kTolNormInf)

    # Translate path to origin
    origin = path[0, :]
    path = path - np.ones((data_size+1, 1)) * origin

    # Get initial curvature guess
    arclength, kappa = initial_guess( path )
    kappa = kappa[0:len(kappa)-1]

    # Pre-solve for improved accuracy
    rot_matrix, kappa, magnitude = pre_solve( path, kappa, arclength, kappa_max )
    # print(kappa)
    print('Sparsity solver')
    timer = time.time()

    # Calculate second derivative matrix
    D = np.eye(data_size)
    D = np.linalg.lstsq(np.diag(magnitude[0:len(magnitude)-1]), (D[0:len(D)-1, :] - D[1:len(D), :]))[0]
    D = np.linalg.lstsq(np.diag(magnitude[0:len(magnitude)-2]), (D[0:len(D)-1, :] - D[1:len(D), :]))[0]
    # print(D)

    kappa_out = kappa
    weight = np.ones(data_size - 2)

    # Execute Sequential Linear Programming
    exit_success = False

    for i in range(0,kMaxIter):#
        timer = time.time()
        # Create a new model
        m = Model("lp")

        # Construct the optimization problem apriori to improve performance
        # Variables
        kappa_v = m.addVars(data_size, lb=-GRB.INFINITY, ub=GRB.INFINITY)       # Optimization variable
        theta_k = m.addVars(data_size, lb=-GRB.INFINITY, ub=GRB.INFINITY)       # (S * (kappa - kappa_lin))
        x_k = m.addVars(data_size+1, lb=-GRB.INFINITY, ub=GRB.INFINITY)         # (S * cos(theta))
        y_k = m.addVars(data_size+1, lb=-GRB.INFINITY, ub=GRB.INFINITY)         # (S * sin(theta))
        norm_slack = m.addVars(data_size-2, lb=-GRB.INFINITY, ub=GRB.INFINITY)  # Slack for norm approximation
        path_k = m.addVars(data_size+1,2, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        x_error = m.addVars(data_size+1,2, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        y_error = m.addVars(data_size+1,2, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        D_kappa_v = m.addVars(data_size-2, lb=-GRB.INFINITY, ub=GRB.INFINITY)

        # Initialize auxiliary variables
        kappa_lin = copy.copy(kappa_out)
        theta_sk = np.cumsum(magnitude * kappa_lin)
        cos_sk = np.cos(theta_sk)
        sin_sk = np.sin(theta_sk)

        #Update weights

        weight = 1 / (weight * ( np.abs(D.dot( kappa_lin )) ) + 1e-6)
        weight = weight * (data_size - 2) / np.sum(weight)


        # Construct point coordinates using first order Taylor expansion
        for p in range(data_size+1):
            path_k[p,0] = rot_matrix[0,:].dot(([x_k[p],y_k[p]]))
            path_k[p,1] = rot_matrix[1,:].dot(([x_k[p],y_k[p]]))

        # Define error variables
        for p in range(data_size+1):
            x_error[p] = path_k[p,0] - path[p,0]
            y_error[p] = path_k[p,1] - path[p,1]

        # Define curvature second derivative variable
        for p in range(data_size-2):
            sum_D_kappa_v = 0
            for q in range(p,p+3):
                sum_D_kappa_v = sum_D_kappa_v + D[p,q]*kappa_v[q]
            D_kappa_v[p] = sum_D_kappa_v

        # Set optimizer object properties
        # Note: We do not care about the first point since it is fixed
        # obj = 0
        # for p in range(data_size - 2):
        #     obj = obj + norm_slack[p]*weight[p]
        obj = 0
        for p in range(data_size - 2):
            obj = obj + kappa_v[p]*1000  
            
        m.setObjective(obj, GRB.MINIMIZE)


        m.addConstrs( ( -norm_slack[p] <= D_kappa_v[p] ) for p in range(data_size-2) )
        m.addConstrs( ( D_kappa_v[p] <= norm_slack[p] ) for p in range(data_size-2) )
        m.addConstrs( ( -max_deviation <= x_error[p] ) for p in range(data_size+1) )
        m.addConstrs( ( x_error[p] <= max_deviation ) for p in range(data_size+1) )
        m.addConstrs( ( -max_deviation <= y_error[p] ) for p in range(data_size+1) )
        m.addConstrs( ( y_error[p] <= max_deviation ) for p in range(data_size+1) )
        m.addConstr( theta_k[0] == magnitude[0] * (kappa_v[0] - kappa_lin[0]) )
        m.addConstrs( theta_k[p]-theta_k[p-1] == magnitude[p] * (kappa_v[p] - kappa_lin[p]) for p in range(1,data_size) )
        m.addConstrs( x_k[p+1] - x_k[p] == magnitude[p] * (cos_sk[p] - sin_sk[p] * theta_k[p]) for p in range(0, data_size) )
        m.addConstrs( y_k[p+1] - y_k[p] == magnitude[p] * (sin_sk[p] + cos_sk[p] * theta_k[p]) for p in range(0, data_size) )
        m.addConstrs( -kMaxDeltaKappa <= kappa_v[p] - kappa_lin[p] for p in range(0, data_size) )
        m.addConstrs( kappa_v[p] - kappa_lin[p] <= kMaxDeltaKappa for p in range(0, data_size) )

        # If is path closed, enforce continuity constraints
        if is_closed:
            m.addConstr( kappa_v[0] == kappa_v[len(kappa_v)] )
            m.addConstrs( ( path_k[0,p] == path_k[len(path_k),p] ) for p in range(2))

        m.Params.OutputFlag = 0

        # Call the solver
        m.optimize()

        for p in range(data_size):
                kappa_out[p] = kappa_v[p].x

        error = np.max(abs(kappa_out - kappa_lin))
        print('    Iteration ' + str(i+1) + ' - Error: ' + str(error) + ' Elapsed: ' + \
              str((time.time()-timer) * 100000.0 / 100.0) + ' miliseconds.')

        # Check if error is sufficiently small
        if error <= kTolNormInf:
            exit_success = True
            print('    Completed - Error tolerance met.')
            break

    if exit_success == False:
        print('    Completed - Exceeded number of iterations.')


    # Calculate the last curvature
    last_kappa = kappa_out[data_size-1] + \
                 np.diff(kappa_out[data_size-2:data_size])/ \
                 arclength[data_size-2] * arclength[data_size-1]
    kappa_out = np.concatenate(( kappa_out, last_kappa ))

    # Calculate the sharpness
    dkappa_out = np.diff(kappa_out) / np.diff(arclength)
    last_dkappa_out = ([dkappa_out[len(dkappa_out)-1]])
    dkappa_out = np.concatenate(( dkappa_out, last_dkappa_out ))

    # Initialize auxiliary variables
    theta = np.cumsum(magnitude * kappa_out[0:len(kappa_out)-1])
    x = np.cumsum(magnitude * np.cos(theta))  # Equivalent to: S * cos(theta);
    y = np.cumsum(magnitude * np.sin(theta))  # Equivalent to: S * sin(theta);

    r_ang = math.atan2(np.diff(path[0:2,1]), np.diff(path[0:2,0])) - math.atan2(y[0], x[0])


    x = np.concatenate((np.zeros(1), x), axis=0)
    y = np.concatenate((np.zeros(1), y), axis=0)

    pos = (rot_matrix.dot(np.array([x, y]))).transpose() + \
        np.ones((data_size+1, 1)) * origin

    theta = np.concatenate( (np.zeros(1), theta) ) + \
        np.ones(data_size+1) * r_ang

    # theta = np.zeros(len(theta))
    # theta = np.arctan2( np.diff(y),np.diff(x) )

    # theta = np.arctan2( np.diff(pos[:,1]),np.diff(pos[:,0]) )
    for p in range(len(theta)-1):
        theta[p] = np.arctan2(pos[p+1,1]-pos[p,1], pos[p+1,0]-pos[p,0])
        theta[p] = norm_ang(theta[p])

    return pos, theta, arclength, kappa_out, dkappa_out

def low_band_filter( path ):

    M = copy.copy( path )

    #Eliminate outliears
    #Ssome coodinates are multiplied by 1000. We want to correct them
    # x = M[:,0]/0.0002e9
    # y = M[:,1]/0.0076e9
    #
    # n = len(x);
    # for k in range(0,n):
    #     if x[k] > 10:
    #         M[k,0] = M[k,0]/1000
    #     if y[k] > 10:
    #         M[k,1] = M[k,1]/1000
    #
    # x = M[:,0]
    # y = M[:,1]
    # m = 0
    # k = 0
    # while m < n-1:
    #     if x[m+1] == x[m] and y[m+1] == y[m]:
    #          m = m + 1
    #          continue
    #     M[k,:] = M[m,:]
    #     m = m + 1
    #     k = k + 1

    #Apply a low pass band filter to improve curvature quality
    b,a = signal.butter(6,0.1)#2,0.025
    x = signal.filtfilt(b,a,M[:,0])
    y = signal.filtfilt(b,a,M[:,1])
    gps_data = np.zeros((len(x),2))
    gps_data[:,0] = x
    gps_data[:,1] = y
    return gps_data

def initial_guess( path ):
# INITIAL_GUESS Calculate curvature and arclength based on point locations
#
#    Input(s):
#    (1) path - X and Y position of points, where each line is a (X, Y) pair.
#
#    Output(s):
#    (1) kappa     - Column vector with curvatures for each point;
#    (2) arclength - Column vector with arclength from first point to each point.
#
#    Author(s):
#    (1) Júnior A. R. Da Silva
#    (2) Carlos M. Massera
#
#    Copyright owned by Universidade de São Paulo

    # Get vector between adjacent points
    vector = np.diff(path[:, 0:2], axis=0)

    # Get heading and magnitude of path vectors
    theta = np.arctan2(vector[:,1], vector[:,0])
    magnitude = np.sqrt(((vector[:,0]**2 + vector[:,1]**2)))

    # Get heading variation
    dtheta = np.diff(theta);

    # Clip between -pi and pi
    dtheta = np.mod(dtheta + math.pi, 2 * math.pi) - math.pi

    # Calculate curvature
    kappa_mag = np.sqrt(magnitude[0:len(magnitude)-1] * magnitude[1:len(magnitude)])
    kappa = 2 * np.sin(dtheta / 2) / kappa_mag

    # Calculate arc length
    arclength = np.concatenate(( [0], np.cumsum(magnitude) ))

    # Initial and end curvature calculation
    #     Initial: Solve for kappa and dkappa using 2nd and 3rd points
    A = ([1, 0],\
         [1, magnitude[1]])
    b = kappa[0:2]
    kappa_1 = np.array([1, -magnitude[0]]).dot(np.linalg.lstsq(A,b)[0])

    #     Final: Solve for kappa and dkappa using the two last available points
    A = ([1, -magnitude[len(magnitude)-2]],\
         [1, 0])
    b = kappa[len(kappa)-2:len(kappa)]
    kappa_end = np.array([1, magnitude[len(magnitude)-1]]).dot( np.linalg.lstsq(A,b)[0])

    #     Concatenate them into one vector
    kappa = np.concatenate(( ([kappa_1]), kappa, ([kappa_end]) ))

    return arclength, kappa

def norm_ang( theta ):
    if theta > math.pi:
        theta_n = theta - 2*math.pi
    elif theta < -math.pi:
        theta_n = 2*math.pi + theta
    else:
        theta_n = theta
    return theta_n

def generate_trajectory( pos, kappa_max ):
    # Calculate the traveled distance and the curvature
    max_deviation = 0.0
    pos, theta, arclength, kappa_out, dkappa_out = optimize_curvature( path, max_deviation, kappa_max )
    fx = interp1d(arclength, pos[:,0], kind='cubic')
    fy = interp1d(arclength, pos[:,1], kind='cubic')

    new_arclength = np.arange(0, arclength[-1], 0.5)
    pos = np.zeros((len(new_arclength), 2))
    pos[:,0] = fx(new_arclength)
    pos[:,1] = fy(new_arclength)

    pos = low_band_filter( pos )
    pos = pos[0:-250,:]


    arclength, kappa_out = initial_guess( pos )

    # Calculate the sharpness
    dkappa_out = np.diff(kappa_out) / np.diff(arclength)
    last_dkappa_out = ([dkappa_out[len(dkappa_out)-1]])
    dkappa_out = np.concatenate(( dkappa_out, last_dkappa_out ))

    # Calculate heading
    theta = np.zeros(len(arclength))
    for p in range(len(theta)-1):
        theta[p] = np.arctan2(pos[p+1,1]-pos[p,1], pos[p+1,0]-pos[p,0])
        theta[p] = norm_ang(theta[p])

    #Make velocity equal to zero since we will not use it
    v = 7*np.ones(len(arclength))

    waypoints = np.zeros((len(pos),7))
    waypoints[:,0] = pos[:,0]
    waypoints[:,1] = pos[:,1]
    waypoints[:,2] = kappa_out
    waypoints[:,3] = theta
    waypoints[:,4] = arclength
    waypoints[:,5] = v
    waypoints[:,6] = dkappa_out

    return waypoints

# with open('skoods_cup_track_limits.npy', 'rb') as f:
#     internal_waypoints_x = np.load(f)
#     internal_waypoints_y = np.load(f)
#     external_waypoints_x = np.load(f)
#     external_waypoints_y = np.load(f)

# plt.plot(internal_waypoints_x, internal_waypoints_y)
# plt.plot(external_waypoints_x, external_waypoints_y, 'r')
plt.axis('equal')
# xy_coordinates = np.load('xy_coordinates.npy')
# xy_coordinates = np.load('xy_coordinates_clot.npy')
xy_coordinates = genfromtxt(os.path.join(output_dir, 'waypoints.csv'), delimiter=',')

plt.plot(xy_coordinates[:,0], xy_coordinates[:,1],'k+')
# n = 40
# plt.plot(xy_coordinates[n,0], xy_coordinates[n,1],'go')
# xy_coordinates = np.delete(xy_coordinates, n, axis=0)
# plt.plot(xy_coordinates[:,0], xy_coordinates[:,1],'go')
# plt.show()
# xy_coordinates = plt.ginput(n=-1, timeout=-1, show_clicks=True)
# file = 'xy_coordinates_clot.npy'
# np.save(file, xy_coordinates)

#Interpolate points
arclength, kappa = initial_guess( xy_coordinates )
fx = interp1d(arclength, xy_coordinates[:,0], kind='cubic')
fy = interp1d(arclength, xy_coordinates[:,1], kind='cubic')

new_arclength = np.arange(0, arclength[-1], 1.0)
new_x = fx(new_arclength)
new_y = fy(new_arclength)

print(len(new_x),len(new_y))
plt.plot(new_x, new_y,'y')
# plt.show()


plt.figure()
plt.plot(arclength, kappa)
# plt.show()

#Apply low band filter
path_interp = np.concatenate( ( np.vstack(new_x), np.vstack(new_y) ), axis=1 )
path = low_band_filter( path_interp )
# path = path[0:-55,:]
plt.figure(1)
plt.plot(path[:,0], path[:,1],'b')

plt.figure(2)
arclength, kappa = initial_guess( path )
plt.plot(arclength, kappa, 'r')
# plt.show()

kappa_max = 1.0
waypoints = generate_trajectory( path, kappa_max )
file = 'waypoints.npy'
np.save(file, waypoints)


# #UNCOMMENT BELOW FOR GRAPHS
# plt.figure(1)
# plt.plot(waypoints[:,0], waypoints[:,1],'g')

# plt.figure(2)
# plt.plot(waypoints[:,4], waypoints[:,2] ,'g')
# plt.show()

######CANNOT BE USED AFTER SPARSE COORDS
# np.savetxt('../output/test_clothoid_result.csv', waypoints, delimiter=',', fmt='%s')
######


####sparse coords
xy_coordinates = np.vstack((new_x, new_y)).T

# Save the array to a CSV file
np.savetxt(os.path.join(output_dir, 'clothoid_fit_xy_coordinates.csv'), xy_coordinates, delimiter=',')