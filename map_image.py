import os
import requests
from PIL import Image, ImageDraw
from io import BytesIO
import math
import csv

# Function to load API credentials from a file
def load_naver_api_credentials(file_path):
    credentials = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(' = ')
            credentials[key] = value.strip("'")  # Remove the surrounding quotes
    return credentials

# Function to crop the image to a specified size in meters centered on the image's center
def crop_image(image, latitude, zoom, crop_size_meters):
    # Calculate the center of the image in pixels
    center_x = image.width // 2
    center_y = image.height // 2
    
    # Calculate meters per pixel
    meters_per_pixel = (40075016.686  * math.cos(math.radians(latitude))) / (2 ** (zoom+1+8))
    
    # Calculate the pixel dimensions for the specified crop size in meters
    crop_size_pixels = int(crop_size_meters / meters_per_pixel)
    print(crop_size_pixels)
    
    # Calculate the bounding box for cropping
    left = center_x - (crop_size_pixels // 2)
    top = center_y - (crop_size_pixels // 2)
    right = center_x + (crop_size_pixels // 2)
    bottom = center_y + (crop_size_pixels // 2)
    
    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image

# Function to plot multiple coordinates on an image
def plot_coordinates_on_image(image, base_latitude, base_longitude, coordinates, zoom):
    # Calculate meters per pixel
    meters_per_pixel = (40075016.686  * math.cos(math.radians(latitude))) / (2 ** (zoom+1+8))
    
    # Calculate the center of the image in pixels
    center_x = image.width // 2
    center_y = image.height // 2
    
    # Draw on the image
    draw = ImageDraw.Draw(image)
    
    for coord in coordinates:
        target_latitude, target_longitude = coord
        
        # Calculate the distance in meters
        delta_lat = (target_latitude - base_latitude) * 111320  # meters per degree latitude
        delta_lon = (target_longitude - base_longitude) * (111320 * math.cos(math.radians(base_latitude)))  # meters per degree longitude adjusted by latitude
        
        # Convert the distance to pixel offsets
        pixel_x_offset = int(delta_lon / meters_per_pixel)
        pixel_y_offset = int(-delta_lat / meters_per_pixel)  # Negative because latitude decreases as you go north
        
        # Convert the offsets to actual pixel coordinates on the image
        point_x = center_x + pixel_x_offset
        point_y = center_y + pixel_y_offset
        
        # Draw a 5-pixel circle at the specified coordinates
        draw.ellipse((point_x - 2, point_y - 2, point_x + 2, point_y + 2), fill='blue')
    
    return image

# Function to read coordinates from a CSV file
def read_coordinates_from_csv(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            latitude, longitude = map(float, row)
            coordinates.append((latitude, longitude))
    return coordinates

# Load credentials
api_path = os.path.join(os.path.dirname(__file__), '../config/naver_api.txt')
credentials = load_naver_api_credentials(api_path)

# Access the CLIENT_ID and CLIENT_SECRET
CLIENT_ID = credentials['CLIENT_ID']
CLIENT_SECRET = credentials['CLIENT_SECRET']

# Coordinates for the center of the image
latitude = 35.9466289
longitude = 126.5909605

# Map parameters
zoom = 18  # Adjust the zoom level as needed
maptype = 'satellite'  # or 'basic' for normal map view
size = '2000x2000'  # Image size

# Naver Static Map API endpoint
url = f"https://naveropenapi.apigw.ntruss.com/map-static/v2/raster?center={longitude},{latitude}&level={zoom}&w={size.split('x')[0]}&h={size.split('x')[1]}&maptype={maptype}"

# Headers with your credentials
headers = {
    'X-NCP-APIGW-API-KEY-ID': CLIENT_ID,
    'X-NCP-APIGW-API-KEY': CLIENT_SECRET
}

# Send the request to Naver Maps Static API
response = requests.get(url, headers=headers)

# Define the relative path to the output directory
output_dir = os.path.join(os.path.dirname(__file__), './output')
os.makedirs(output_dir, exist_ok=True)

# Check if the request was successful
if response.status_code == 200:
    # Open the image
    image = Image.open(BytesIO(response.content))
    
    # Crop the image to 440m x 440m centered on the origin
    cropped_image = crop_image(image, latitude, zoom, 440)
    
    # Save and show the cropped image
    cropped_image.save(os.path.join(output_dir, 'naver_satellite_image_cropped_440x440.png'))
    cropped_image.show()
    
    # Path to the CSV file with coordinates
    csv_file_path = os.path.join(os.path.dirname(__file__), './coordinates.csv')
    
    # Read coordinates from the CSV file
    coordinates = read_coordinates_from_csv(csv_file_path)
    
    # Plot the coordinates on the cropped image
    image_with_points = plot_coordinates_on_image(cropped_image, latitude, longitude, coordinates, zoom)
    
    # Save and show the cropped image with the points
    image_with_points.save(os.path.join(output_dir, 'naver_satellite_image_with_points.png'))
    image_with_points.show()

else:
    print("Error fetching the satellite image:", response.status_code, response.text)
