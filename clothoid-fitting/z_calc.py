import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Step 1: Load the Generated 2D Clothoidal Path
xy_coordinates = np.genfromtxt('new_xy_coordinates.csv', delimiter=',', skip_header=1)
x = xy_coordinates[:, 0]
y = xy_coordinates[:, 1]

# Step 2: Load original waypoints for Z values
waypoints = np.genfromtxt('waypoints.csv', delimiter=',', skip_header=1)
waypoint_x = waypoints[:, 0]
waypoint_y = waypoints[:, 1]
waypoint_z = waypoints[:, 2]

# Step 3: Calculate the Arc Length of the 2D Path
arc_length = np.zeros(len(x))
arc_length[1:] = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

# Step 4: Calculate the Arc Length at Waypoints Directly
waypoint_arc_lengths = np.zeros(len(waypoints))
waypoint_arc_lengths[1:] = np.cumsum(np.sqrt(np.diff(waypoint_x)**2 + np.diff(waypoint_y)**2))

# Step 5: Interpolate the Z Values Over the Entire Path
z = np.interp(arc_length, waypoint_arc_lengths, waypoint_z)

# Step 6: Calculate the Slope of Z
slope = np.gradient(z, arc_length)

# Step 7: Toggle for Color Mode (choose 'altitude' or 'slope')
color_mode = 'slope'  # Change this to 'slope' to color by slope

if color_mode == 'altitude':
    color_data = z
    color_label = 'Altitude (meters)'
else:
    color_data = slope
    color_label = 'Slope (dz/ds)'

# Step 8: Combine X, Y, and Z into a 3D Path
xyz_path = np.vstack((x, y, z)).T

# Step 9: Save the 3D Path
np.savetxt('clothoidal_path_3d.csv', xyz_path, delimiter=',', header='x,y,z', comments='', fmt='%.6f')

print("3D clothoidal path saved to 'clothoidal_path_3d.csv'")

# Step 10: Plot the 3D Clothoidal Path in the X,Y Plane with Color Dependent on Altitude or Slope
plt.figure(figsize=(12, 8))
scatter = plt.scatter(x, y, c=color_data, cmap='viridis', s=10)

# Mark the original waypoints with black crosses
plt.scatter(waypoint_x, waypoint_y, color='black', marker='x', s=100, label='Waypoints')

# Add a color bar to show the altitude or slope
cbar = plt.colorbar(scatter)
cbar.set_label(color_label, rotation=270, labelpad=15)

plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.title('3D Clothoidal Path in X,Y Plane with ' + color_label + ' Coloring')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
