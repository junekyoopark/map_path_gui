import sys
import os
import csv
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
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

latitude = 35.9466289
longitude = 126.5909605
zoom = 18
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
        
        self.setWindowTitle("Map Click Coordinates")
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

        # Load button
        self.load_button = QPushButton("Load Coordinates", self)
        self.load_button.clicked.connect(self.load_coordinates)
        layout.addWidget(self.load_button)

        # Save As button
        self.save_button = QPushButton("Save As", self)
        self.save_button.clicked.connect(self.save_as)
        layout.addWidget(self.save_button)

        # Save in Meters button
        self.save_meters_button = QPushButton("Save Waypoints in Meters", self)
        self.save_meters_button.clicked.connect(self.save_waypoints_in_meters)
        layout.addWidget(self.save_meters_button)

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
        delta_lat = -(-y + (display_height/2)) * self.meters_per_pixel / 111320
        delta_lon = (x - (display_width/2)) * self.meters_per_pixel / (111320 * math.cos(math.radians(self.latitude)))

        clicked_lat = self.latitude - delta_lat
        clicked_lon = self.longitude + delta_lon

        # Insert the new point between segment_index and segment_index + 1
        self.click_history.insert(segment_index + 1, (x, y, clicked_lat, clicked_lon))
        self.action_history.append(('insert', segment_index + 1))  # Record the action
        self.redraw_points()

    def add_waypoint(self, x, y):
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
        self.click_history.append((x, y, clicked_lat, clicked_lon))
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
                        if len(row) == 2:
                            lat, lon = map(float, row)
                            x = (lon - self.longitude) * (111320 * math.cos(math.radians(self.latitude))) / self.meters_per_pixel + (display_width / 2)
                            y = -(lat - self.latitude) * 111320 / self.meters_per_pixel + (display_height / 2)
                            self.click_history.append((int(x), int(y), lat, lon))
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
                        # Calculate distances in meters from the origin (latitude, longitude)
                        delta_lat_meters = -(lat - self.latitude) * 111320
                        delta_lon_meters = (lon - self.longitude) * 111320 * math.cos(math.radians(self.latitude))
                        writer.writerow([delta_lat_meters, delta_lon_meters])
                print("Waypoints in meters saved successfully.")
            except Exception as e:
                print("Failed to save waypoints in meters:", str(e))

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
