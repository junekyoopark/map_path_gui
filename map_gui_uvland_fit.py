import sys
import os
import subprocess
import csv
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt
from PIL import Image
import requests
from io import BytesIO
import math

width = 2000
height = 2000
display_width = 1000
display_height = 1000

latitude = 36.661078
longitude = 126.342193
zoom = 17
maptype = 'satellite'

def load_naver_api_credentials(file_path):
    credentials = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(' = ')
            credentials[key] = value.strip("'")
    return credentials

def fetch_map_image(latitude, longitude, zoom, maptype, size, CLIENT_ID, CLIENT_SECRET):
    url = f"https://naveropenapi.apigw.ntruss.com/map-static/v2/raster?center={longitude},{latitude}&level={zoom}&w={size.split('x')[0]}&h={size.split('x')[1]}&maptype={maptype}"
    headers = {
        'X-NCP-APIGW-API-KEY-ID': CLIENT_ID,
        'X-NCP-APIGW-API-KEY': CLIENT_SECRET
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print("Error fetching the satellite image:", response.status_code, response.text)
        sys.exit(1)

class MapClickApp(QMainWindow):
    def __init__(self, image, latitude, longitude, zoom, csv_file_path):
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.zoom = zoom
        self.csv_file_path = csv_file_path
        self.meters_per_pixel = (40075016.686 * math.cos(math.radians(latitude))) / (2 ** (zoom + 8))
        
        self.setWindowTitle("GUI Path Generator (UVLand)")
        self.setGeometry(100, 100, display_width, display_height)

        self.base_image = image.resize((display_width, display_height), Image.ANTIALIAS)
        self.base_image = self.base_image.convert("RGB")
        data = self.base_image.tobytes("raw", "RGB")
        qimage = QImage(data, display_width, display_height, QImage.Format_RGB888)
        self.base_pixmap = QPixmap.fromImage(qimage)

        self.label = QLabel(self)
        self.label.setPixmap(self.base_pixmap.copy())
        self.label.mousePressEvent = self.get_pos

        self.undo_button = QPushButton("Undo", self)
        self.undo_button.clicked.connect(self.undo_last_action)
        layout = QVBoxLayout(self)
        layout.addWidget(self.undo_button)
        layout.addWidget(self.label)

        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(layout)
        
        self.click_history = []

        # Additional history stack to track actions
        self.action_history = []
        self.csv_loaded = False  # Track if a CSV has been loaded

        # Horizontal layout for load, save, and load in meters buttons
        button_layout = QHBoxLayout()

        # Load button
        self.load_button = QPushButton("Load WP (Lat/Lon)", self)
        self.load_button.clicked.connect(self.load_coordinates)
        button_layout.addWidget(self.load_button)

        # Save As button
        self.save_button = QPushButton("Save WP (Lat/Lon)", self)
        self.save_button.clicked.connect(self.save_as)
        button_layout.addWidget(self.save_button)

        # Save in Meters button
        self.save_meters_button = QPushButton("Save WP (m)", self)
        self.save_meters_button.clicked.connect(self.save_waypoints_in_meters)
        button_layout.addWidget(self.save_meters_button)

        # Load in Meters button
        self.load_meters_button = QPushButton("Load WP (m)", self)
        self.load_meters_button.clicked.connect(self.load_waypoints_in_meters)
        button_layout.addWidget(self.load_meters_button)

        # Clothoid fit button
        self.fit_button = QPushButton("Clothoid", self)
        self.fit_button.clicked.connect(self.fit_clothoid)
        button_layout.addWidget(self.fit_button)

        # Z smooth button
        self.run_z_calc_meters_button = QPushButton('Smooth Z (Meters Source)', self)
        self.run_z_calc_meters_button.clicked.connect(self.run_z_calc_meters)
        button_layout.addWidget(self.run_z_calc_meters_button)

        self.run_z_calc_lat_long_button = QPushButton('Smooth Z (Lat/Lon Source)', self)
        self.run_z_calc_lat_long_button.clicked.connect(self.run_z_calc_lat_long)
        button_layout.addWidget(self.run_z_calc_lat_long_button)

        layout.addLayout(button_layout)

        self.coordinate_mode = 'latlon'  # or 'meters'

    def get_pos(self, event):
        x, y = event.pos().x(), event.pos().y()
        min_dist = float('inf')
        nearest_segment = None

        if self.csv_loaded:  # Only allow insertion if CSV is loaded
            min_dist = float('inf')
            nearest_segment = None

            for i in range(len(self.click_history) - 1):
                p1 = self.click_history[i]
                p2 = self.click_history[i + 1]
                dist = self.distance_to_segment(x, y, p1[0], p1[1], p2[0], p2[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_segment = i

            if nearest_segment is not None and min_dist < 10000:  # 10 pixels as threshold
                self.insert_waypoint(nearest_segment, x, y)
            else:
                self.add_waypoint(x, y)

        else:
            # If no CSV is loaded, just add the waypoint normally
            self.add_waypoint(x, y)

    def distance_to_segment(self, px, py, x1, y1, x2, y2):
        """Calculate the minimum distance from a point to a line segment."""
        line_mag = math.hypot(x2 - x1, y2 - y1)
        if line_mag < 1e-10:
            return math.hypot(px - x1, py - y1)
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
        if u < 0:
            closest_point = (x1, y1)
        elif u > 1:
            closest_point = (x2, y2)
        else:
            closest_point = (x1 + u * (x2 - x1), y1 + u * (y2 - y1))
        return math.hypot(px - closest_point[0], py - closest_point[1])

    def insert_waypoint(self, segment_index, x, y):
        if self.coordinate_mode == 'latlon':
            delta_lat = -(-y + (display_height/2)) * self.meters_per_pixel / 111320
            delta_lon = (x - (display_width/2)) * self.meters_per_pixel / (111320 * math.cos(math.radians(self.latitude)))

            clicked_lat = self.latitude - delta_lat
            clicked_lon = self.longitude + delta_lon

        elif self.coordinate_mode == 'meters':
            delta_lon = (x - (display_height / 2)) * self.meters_per_pixel
            delta_lat = -(-y + (display_width / 2)) * self.meters_per_pixel

            clicked_lat = self.latitude - delta_lat / 111320
            clicked_lon = self.longitude + delta_lon / (111320 * math.cos(math.radians(self.latitude)))

        # Insert the new point between segment_index and segment_index + 1
        self.click_history.insert(segment_index + 1, (x, y, clicked_lat, clicked_lon))
        self.action_history.append(('insert', segment_index + 1))  # Record the action
        self.redraw_points()

    def add_waypoint(self, x, y):
        # if self.coordinate_mode == 'latlon':
        #     delta_lat = -(-y + (display_height/2)) * self.meters_per_pixel / 111320
        #     delta_lon = (x - (display_width/2)) * self.meters_per_pixel / (111320 * math.cos(math.radians(self.latitude)))

        #     clicked_lat = self.latitude - delta_lat
        #     clicked_lon = self.longitude + delta_lon
        #     self.save_to_csv(clicked_lat, clicked_lon)

        #     self.click_history.append((x, y, clicked_lat, clicked_lon))

        # elif self.coordinate_mode == 'meters':
        #     delta_lat = -(-y + (display_height / 2)) * self.meters_per_pixel
        #     delta_lon = (x - (display_width / 2)) * self.meters_per_pixel

        #     lat = self.latitude + delta_lat / 111320
        #     lon = self.longitude + delta_lon / (111320 * math.cos(math.radians(self.latitude)))

        #     self.click_history.append((x, y, lat, lon))

        delta_lat = -(-y + (display_height/2)) * self.meters_per_pixel / 111320
        delta_lon = (x - (display_width/2)) * self.meters_per_pixel / (111320 * math.cos(math.radians(self.latitude)))

        clicked_lat = self.latitude - delta_lat
        clicked_lon = self.longitude + delta_lon
        self.save_to_csv(clicked_lat, clicked_lon)

        painter = QPainter(self.label.pixmap())
        painter.setPen(QPen(Qt.red, 5))
        painter.drawPoint(x, y)
        painter.end()
        self.label.update()
        self.action_history.append(('add', len(self.click_history) - 1))  # Record the action

    def save_to_csv(self, lat, lon):
        with open(self.csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([lat, lon])

    def undo_last_action(self):
        if not self.action_history:
            return
        last_action, index = self.action_history.pop()
        if last_action in ['insert', 'add']:
            self.click_history.pop(index)
        self.redraw_points()

    def redraw_points(self):
        self.label.setPixmap(self.base_pixmap.copy())
        painter = QPainter(self.label.pixmap())
        painter.setPen(QPen(Qt.red, 5))

        for i in range(len(self.click_history) - 1):
            p1 = self.click_history[i]
            p2 = self.click_history[i + 1]
            painter.setPen(QPen(Qt.blue, 1))
            painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        for x, y, lat, lon in self.click_history:
            painter.setPen(QPen(Qt.red, 5))
            painter.drawPoint(x, y)

        painter.setPen(QPen(Qt.red, 5))
        painter.end()
        self.label.update()
        self.update_csv()

    def update_csv(self):
        with open(self.csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for _, _, lat, lon in self.click_history:
                writer.writerow([lat, lon])

    def load_coordinates(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV file", "", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, 'r', newline='') as file:
                    reader = csv.reader(file)
                    self.click_history.clear()
                    for row in reader:
                        if len(row) >= 2:  # Ensure there are at least two columns
                            lat = float(row[0])  # First column as latitude
                            lon = float(row[1])  # Second column as longitude
                            x = (lon - self.longitude) * (111320 * math.cos(math.radians(self.latitude))) / self.meters_per_pixel + (display_width / 2)
                            y = -(lat - self.latitude) * 111320 / self.meters_per_pixel + (display_height / 2)
                            self.click_history.append((int(x), int(y), lat, lon))
                self.coordinate_mode = 'latlon'
                self.redraw_points()
                self.csv_loaded = True  # Set flag to True when CSV is successfully loaded
            except Exception as e:
                print("Failed to load coordinates:", str(e))

    def save_as(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv)", options=options)
        if file_path:
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            try:
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    for _, _, lat, lon in self.click_history:
                        writer.writerow([lat, lon])
                print("File saved successfully.")
            except Exception as e:
                print("Failed to save file:", str(e))


    def save_waypoints_in_meters(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Waypoints in Meters", "", "CSV Files (*.csv)", options=options)
        if file_path:
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            try:
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    
                    for x, y, lat, lon in self.click_history:
                        delta_lat_meters = (lat - self.latitude) * 111320
                        delta_lon_meters = (lon - self.longitude) * 111320 * math.cos(math.radians(self.latitude))
                        writer.writerow([delta_lon_meters, delta_lat_meters])
                        print(str(lat)+","+str(lon))
                    print("Waypoints in meters saved successfully.")
            except Exception as e:
                print("Failed to save waypoints in meters:", str(e))


    def load_waypoints_in_meters(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Meters CSV file", "", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, 'r', newline='') as file:
                    reader = csv.reader(file)
                    self.click_history.clear()
                    for row in reader:
                        if len(row) >= 2:
                            delta_lon_meters = float(row[0])  # First column as latitude
                            delta_lat_meters = float(row[1])  # Second column as longitude
                            lat = self.latitude + (delta_lat_meters / 111320)
                            lon = self.longitude + (delta_lon_meters / (111320 * math.cos(math.radians(self.latitude))))
                            x = (lon - self.longitude) * (111320 * math.cos(math.radians(self.latitude))) / self.meters_per_pixel + (display_width / 2)
                            y = -(lat - self.latitude) * 111320 / self.meters_per_pixel + (display_height / 2)
                            self.click_history.append((int(x), int(y), lat, lon))
                    self.coordinate_mode = 'meters'

                    self.redraw_points()
                    self.csv_loaded = True
                    print("Waypoints in meters loaded successfully.")
            except Exception as e:
                print("Failed to load waypoints in meters:", str(e))

    def fit_clothoid(self):
        # Step 1: Save the current waypoints as a CSV file
        temp_csv_file = os.path.join(os.path.dirname(__file__), 'temp/waypoints.csv')
        with open(temp_csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for _, _, lat, lon in self.click_history:
                delta_lat_meters = (lat - self.latitude) * 111320
                delta_lon_meters = (lon - self.longitude) * 111320 * math.cos(math.radians(self.latitude))
                writer.writerow([delta_lon_meters, delta_lat_meters])

        # Step 2: Run the clothoid-fitting program
        try:
            # Update the script path to the correct location of the clothoid fitting script
            script_path = os.path.join(os.path.dirname(__file__), 'clothoid-fitting/clothoid-fitting-optimization.py')
            subprocess.run(['python3', script_path], check=True)

            # Step 3: Load the new CSV file generated by the clothoid-fitting program
            fitted_csv_file = os.path.join(os.path.dirname(__file__), 'temp/clothoid_fit_xy_coordinates.csv')

            if os.path.exists(fitted_csv_file):
                self.load_waypoints_in_meters_from_file(fitted_csv_file)
            else:
                print("Fitted CSV file not found.")

        except subprocess.CalledProcessError as e:
            print("Error running clothoid fitting:", e)

    def load_waypoints_in_meters_from_file(self, file_path):
        try:
            with open(file_path, 'r', newline='') as file:
                reader = csv.reader(file)
                self.click_history.clear()
                for row in reader:
                    if len(row) >= 2:
                        delta_lon_meters, delta_lat_meters = map(float, row[:2])
                        lat = self.latitude + (delta_lat_meters / 111320)
                        lon = self.longitude + (delta_lon_meters / (111320 * math.cos(math.radians(self.latitude))))
                        x = (lon - self.longitude) * (111320 * math.cos(math.radians(self.latitude))) / self.meters_per_pixel + (display_width / 2)
                        y = -(lat - self.latitude) * 111320 / self.meters_per_pixel + (display_height / 2)
                        self.click_history.append((int(x), int(y), lat, lon))
                self.coordinate_mode = 'meters'
                self.redraw_points()
                self.csv_loaded = True
                print("Waypoints from fitting loaded successfully.")
        except Exception as e:
            print("Failed to load waypoints from fitting:", str(e))

    def run_z_calc_meters(self):
        # Save the waypoints on the screen to 'new_xy_coordinates.csv'
        temp_csv_file = os.path.join(os.path.dirname(__file__), 'temp/clothoid_waypoints.csv')
        with open(temp_csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for _, _, lat, lon in self.click_history:
                delta_lat_meters = (lat - self.latitude) * 111320
                delta_lon_meters = (lon - self.longitude) * 111320 * math.cos(math.radians(self.latitude))
                writer.writerow([delta_lon_meters, delta_lat_meters])
        
        # Open a file dialog to select the original waypoint file
        waypoint_file_path, _ = QFileDialog.getOpenFileName(self, "Select Source Waypoints File", "", "CSV Files (*.csv)")
        
        if waypoint_file_path:
            try:
                # Copy the selected source waypoints file to the working directory
                # Assuming the script expects the file to be named 'z_calc_source_waypoints.csv'
                with open(waypoint_file_path, 'r') as infile:
                    reader = csv.reader(infile)
                    
                    with open('temp/z_calc_source_waypoints.csv', 'w', newline='') as outfile:
                        writer = csv.writer(outfile)
                        for row in reader:
                            writer.writerow(row)
                
                # Step 3: Run the z_calc.py script
                subprocess.run(['python3', 'clothoid-fitting/z_calc.py'], check=True)
                print("Z calculation completed successfully.")
            except Exception as e:
                print(f"An error occurred while processing the waypoints file or running z_calc.py: {e}")
        else:
            print("No waypoints file selected. Z calculation aborted.")

    def run_z_calc_lat_long(self):
        # Save the waypoints on the screen to 'new_xy_coordinates.csv'
        temp_csv_file = os.path.join(os.path.dirname(__file__), 'temp/clothoid_waypoints.csv')
        with open(temp_csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for _, _, lat, lon in self.click_history:
                delta_lat_meters = (lat - self.latitude) * 111320
                delta_lon_meters = (lon - self.longitude) * 111320 * math.cos(math.radians(self.latitude))
                writer.writerow([delta_lon_meters, delta_lat_meters])
        
        # Open a file dialog to select the original waypoint file
        waypoint_file_path, _ = QFileDialog.getOpenFileName(self, "Select Source Waypoints File", "", "CSV Files (*.csv)")
        
        if waypoint_file_path:
            try:
                # Copy and convert the selected source waypoints file to meters and save it to the working directory
                with open(waypoint_file_path, 'r') as infile:
                    reader = csv.reader(infile)
                    
                    with open('temp/z_calc_source_waypoints.csv', 'w', newline='') as outfile:
                        writer = csv.writer(outfile)
                        for row in reader:
                            lat = float(row[0])
                            lon = float(row[1])
                            z = float(row[2])  # Assuming Z value is already in meters

                            # Convert lat/lon to meters
                            delta_lat_meters = (lat - self.latitude) * 111320
                            delta_lon_meters = (lon - self.longitude) * 111320 * math.cos(math.radians(self.latitude))

                            # Write the converted values to the output file
                            writer.writerow([delta_lon_meters, delta_lat_meters, z])
                            
                # Step 3: Run the z_calc.py script
                subprocess.run(['python3', 'clothoid-fitting/z_calc.py'], check=True)
                print("Z calculation completed successfully.")
            except Exception as e:
                print(f"An error occurred while processing the waypoints file or running z_calc.py: {e}")
        else:
            print("No waypoints file selected. Z calculation aborted.")

def main():
    api_path = os.path.join(os.path.dirname(__file__), '../config/naver_api.txt')
    credentials = load_naver_api_credentials(api_path)
    CLIENT_ID = credentials['CLIENT_ID']
    CLIENT_SECRET = credentials['CLIENT_SECRET']
    size = f'{width}x{height}'
    image = fetch_map_image(latitude, longitude, zoom, maptype, size, CLIENT_ID, CLIENT_SECRET)
    csv_file_path = os.path.join(os.path.dirname(__file__), './output/clicked_coordinates.csv')
    app = QApplication(sys.argv)
    window = MapClickApp(image, latitude, longitude, zoom, csv_file_path)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
