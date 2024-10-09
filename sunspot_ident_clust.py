import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import requests
import tarfile
import os
import io
import random
import shutil

# Function to find dark spots and draw circles
def find_dark_spots(image, threshold_value):
    img_np = np.array(image.convert("L"))  # Convert to grayscale
    img_binary = np.where(img_np > threshold_value, 255, 0).astype(np.uint8)

    # Find coordinates of dark spots
    y_coords, x_coords = np.where(img_binary == 0)  # Find remaining black pixels
    return list(zip(x_coords, y_coords))

# Function to download and extract TAR file from a specific URL
def download_and_extract_tar(tar_url):
    try:
        response = requests.get(tar_url)
        response.raise_for_status()  # Raise an error for bad responses

        # Specify a more permanent directory for images
        image_directory = os.path.abspath("solar_images")
        os.makedirs(image_directory, exist_ok=True)

        with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as tar:
            tar.extractall(image_directory)  # Extract to solar_images folder
            
            # List the extracted files for debugging
            extracted_files = tar.getnames()
            st.write("Extracted files:", extracted_files)

        # Return absolute paths for .jpg image files
        return [os.path.join(image_directory, f) for f in extracted_files if f.endswith('.jpg')]
    
    except Exception as e:
        st.error(f"Failed to retrieve or extract TAR file: {e}")
        return []

# Function to clean up temp folder
def cleanup_temp_folder():
    if os.path.exists("solar_images"):
        shutil.rmtree("solar_images")

# Streamlit app layout
st.title("Solar Dark Spot Analysis from Images")

# Initialize session state variables
if 'image_files' not in st.session_state:
    st.session_state.image_files = []
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'dark_spots' not in st.session_state:
    st.session_state.dark_spots = []

# Define the TAR file URL directly
tar_url = "https://github.com/madhan-phy/ML/raw/a7c33130d06525558d75dc1da011372d82daaaad/solar-images/solar_pics.tar.gz"

# Slider for the number of images to fetch
num_images = st.slider("Select number of images to process:", 1, 1000, 1)

if st.button("Fetch Images from GitHub"):
    st.session_state.image_files = download_and_extract_tar(tar_url)
    if st.session_state.image_files:
        # Randomly select the desired number of images
        selected_images = random.sample(st.session_state.image_files, min(num_images, len(st.session_state.image_files)))
        st.session_state.image_files = selected_images
        st.success(f"Fetched {len(st.session_state.image_files)} images from GitHub.")
    else:
        st.error("No images found.")

# Slider for threshold value
threshold_value = st.slider("Select Threshold Value for Dark Spot Detection:", 0, 255, 50)

if st.button("Process Images"):
    if st.session_state.image_files:  # Check if image_files is not empty
        st.session_state.original_image = Image.open(st.session_state.image_files[0])
        st.session_state.dark_spots = find_dark_spots(st.session_state.original_image, threshold_value)
        st.success("Image processing complete.")

# Display the original image with dark spots circled
if st.session_state.original_image is not None:
    # Create a copy of the original image to draw on
    img_with_circles = st.session_state.original_image.copy()
    draw = ImageDraw.Draw(img_with_circles)

    # Draw circles around detected dark spots without altering background
    for (x, y) in st.session_state.dark_spots:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="red", width=1)  # Draw a small circle around the spot

    st.image(img_with_circles, caption="Original Image with Dark Spots", use_column_width=True)

# Option to clean up the folder
if st.button("Clean up image folder"):
    cleanup_temp_folder()
    st.success("Cleaned up the solar_images folder.")
