import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Step 1: Load the Generated 2D Clothoidal Path
xy_coordinates = np.genfromtxt('new_xy_coordinates.csv', delimiter=',', skip_header=1)
x = xy_coordinates[:, 0]
y = xy_coordinates[:, 1]

# Step 2: Load original waypoints for Z values
waypoints = np.genfromtxt('waypoints.csv', delimiter=',', skip_header=1)
z_start = waypoints[0, 2]  # z value of the first waypoint
z_end = waypoints[-1, 2]  # z value of the last waypoint

# Step 3: Calculate the Arc Length of the 2D Path
arc_length = np.zeros(len(x))
arc_length[1:] = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

# Step 4: Define the Z Values
# Define the start and end arc lengths for z interpolation
start_length = arc_length[0] + 10  # 10 meters after the start
end_length = arc_length[-1] - 10  # 10 meters before the end

# Interpolate z values linearly
z = np.interp(arc_length, [arc_length[0], start_length, end_length, arc_length[-1]], [z_start, z_start, z_end, z_end])

# Step 5: Combine X, Y, and Z into a 3D Path
xyz_path = np.vstack((x, y, z)).T

# Step 6: Save the 3D Path
np.savetxt('clothoidal_path_3d.csv', xyz_path, delimiter=',', header='x,y,z', comments='', fmt='%.6f')

print("3D clothoidal path saved to 'clothoidal_path_3d.csv'")

# Step 7: Plot the 3D Clothoidal Path in the X,Y Plane with Z as Color
plt.figure(figsize=(12, 8))
scatter = plt.scatter(x, y, c=z, cmap='viridis', s=10)

# Add a color bar to show the altitude (z value)
cbar = plt.colorbar(scatter)
cbar.set_label('Altitude (meters)', rotation=270, labelpad=15)

plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.title('3D Clothoidal Path in X,Y Plane with Altitude Coloring')
plt.grid(True)
plt.axis('equal')
plt.show()