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
# Scale down the image for display
display_width = 1000
display_height = 1000

latitude = 35.9466289
longitude = 126.5909605
zoom = 18
maptype = 'satellite'

# Function to load API credentials from a file
def load_naver_api_credentials(file_path):
    credentials = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(' = ')
            credentials[key] = value.strip("'")
    return credentials

# Function to fetch the image from Naver Static Map API
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

# PyQt application class
class MapClickApp(QMainWindow):
    def __init__(self, image, latitude, longitude, zoom, csv_file_path):
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.zoom = zoom
        self.csv_file_path = csv_file_path
        self.meters_per_pixel = (40075016.686 * math.cos(math.radians(latitude))) / (2 ** (zoom + 8))
        
        # Set up the user interface
        self.setWindowTitle("Map Click Coordinates")
        self.setGeometry(100, 100, display_width, display_height)

        # Resize and convert the image for display
        self.base_image = image.resize((display_width, display_height), Image.ANTIALIAS)
        self.base_image = self.base_image.convert("RGB")
        data = self.base_image.tobytes("raw", "RGB")
        qimage = QImage(data, display_width, display_height, QImage.Format_RGB888)
        self.base_pixmap = QPixmap.fromImage(qimage)

        # Create label to display the image
        self.label = QLabel(self)
        self.label.setPixmap(self.base_pixmap.copy())
        self.label.mousePressEvent = self.get_pos

        # Add undo button
        self.undo_button = QPushButton("Undo", self)
        self.undo_button.clicked.connect(self.undo_last_action)
        layout = QVBoxLayout(self)
        layout.addWidget(self.undo_button)
        layout.addWidget(self.label)

        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(layout)
        
        # Click history for undo functionality
        self.click_history = []

        # Load waypoints
        self.load_button = QPushButton("Load Coordinates", self)
        self.load_button.clicked.connect(self.load_coordinates)
        layout.addWidget(self.load_button)


    def get_pos(self, event):
        x, y = event.pos().x(), event.pos().y()

        # Center coordinates
        delta_lat = -(-y + (display_height/2)) * self.meters_per_pixel / 111320
        delta_lon = (x - (display_width/2)) * self.meters_per_pixel / (111320 * math.cos(math.radians(self.latitude)))

        clicked_lat = self.latitude - delta_lat
        clicked_lon = self.longitude + delta_lon
        self.save_to_csv(clicked_lat, clicked_lon)

        # Draw on the pixmap
        painter = QPainter(self.label.pixmap())
        painter.setPen(QPen(Qt.red, 5))
        painter.drawPoint(x, y)
        painter.end()
        self.label.update()
        self.click_history.append((x, y, clicked_lat, clicked_lon))

    def save_to_csv(self, lat, lon):
        with open(self.csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([lat, lon])


    def undo_last_action(self):
        if self.click_history:
            self.click_history.pop()  # Remove the last click
            self.redraw_points()

    def redraw_points(self):
        # Reset the pixmap to the original
        self.label.setPixmap(self.base_pixmap.copy())
        painter = QPainter(self.label.pixmap())
        for x, y, lat, lon in self.click_history:
            painter.setPen(QPen(Qt.red, 5))
            painter.drawPoint(x, y)
        painter.end()
        self.label.update()
        self.update_csv()

    def update_csv(self):
        with open(self.csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for _, _, lat, lon in self.click_history:
                writer.writerow([lat, lon])

    def load_coordinates(self):
        try:
            with open(self.csv_file_path, 'r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) == 2:
                        lat, lon = map(float, row)
                        x = (lon - self.longitude) * (111320 * math.cos(math.radians(self.latitude))) / self.meters_per_pixel + (display_width / 2)
                        y = -(lat - self.latitude) * 111320 / self.meters_per_pixel + (display_height / 2)
                        self.click_history.append((x, y, lat, lon))
            self.redraw_points()
        except Exception as e:
            print("Failed to load coordinates:", str(e))


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
