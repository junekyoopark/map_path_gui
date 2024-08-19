import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QFileDialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import subprocess

class WaypointEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.waypoints = None
        self.new_waypoints = []

    def initUI(self):
        self.setWindowTitle('Waypoint Editor')
        
        # Main widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        # Layout
        layout = QVBoxLayout(self.main_widget)
        
        # Matplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Buttons
        self.load_button = QPushButton('Load waypoints.csv', self)
        self.load_button.clicked.connect(self.load_waypoints)
        layout.addWidget(self.load_button)
        
        self.save_button = QPushButton('Save waypoints.csv', self)
        self.save_button.clicked.connect(self.save_waypoints)
        layout.addWidget(self.save_button)
        
        self.run_button = QPushButton('Run Clothoid Fitting and Z Interpolation', self)
        self.run_button.clicked.connect(self.run_clothoid_fitting)
        layout.addWidget(self.run_button)
        
        # Canvas click event
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def load_waypoints(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load waypoints.csv", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.waypoints = pd.read_csv(file_name).values
            self.new_waypoints = []
            self.plot_waypoints()

    def save_waypoints(self):
        if self.waypoints is not None:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save waypoints.csv", "", "CSV Files (*.csv);;All Files (*)", options=options)
            if file_name:
                all_waypoints = np.vstack((self.waypoints, self.new_waypoints))
                pd.DataFrame(all_waypoints, columns=['x', 'y', 'z']).to_csv(file_name, index=False)
    
    def plot_waypoints(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self.waypoints is not None:
            ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], c='black', marker='x', label='Original Waypoints')
        if len(self.new_waypoints) > 0:
            new_waypoints_arr = np.array(self.new_waypoints)
            ax.scatter(new_waypoints_arr[:, 0], new_waypoints_arr[:, 1], c='red', marker='o', label='Synthetic Waypoints')
        ax.legend()
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.set_title('Waypoint Editor')
        ax.grid(True)
        self.canvas.draw()
    
    def on_click(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            z = 0  # Assign a default Z value for synthetic waypoints
            self.new_waypoints.append([x, y, z])
            self.plot_waypoints()

    def run_clothoid_fitting(self):
        if self.waypoints is not None:
            # Combine original and synthetic waypoints
            all_waypoints = np.vstack((self.waypoints, self.new_waypoints))
            
            # Sort waypoints by x and y for simplicity in this example (may need actual arc length sorting)
            all_waypoints = all_waypoints[np.lexsort((all_waypoints[:, 1], all_waypoints[:, 0]))]

            # Save the combined waypoints to a temporary file
            temp_waypoints_file = 'temp_waypoints.csv'
            pd.DataFrame(all_waypoints, columns=['x', 'y', 'z']).to_csv(temp_waypoints_file, index=False)

            # Run the clothoid-fitting script
            self.run_external_script('clothoid-fitting-optimization.py', temp_waypoints_file)

            # Run the Z interpolation script
            self.run_external_script('z_calc.py', temp_waypoints_file)

            # Reload the waypoints after processing
            self.waypoints = pd.read_csv(temp_waypoints_file).values
            self.new_waypoints = []  # Reset synthetic waypoints
            self.plot_waypoints()

    def run_external_script(self, script_name, waypoints_file):
        """Runs an external Python script with the provided waypoints file."""
        try:
            subprocess.run(['python', script_name, waypoints_file], check=True)
            print(f"Successfully ran {script_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_name}: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WaypointEditor()
    ex.show()
    sys.exit(app.exec_())
