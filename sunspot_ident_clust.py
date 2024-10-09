import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw
import requests
import tarfile
import os
import io
import random  # Import random for image selection
import shutil

# Function to process images and mark dark spots
def process_images(image_files, threshold_value):
    processed_images = []
    dark_spots = []
    
    total_images = len(image_files)
    progress_bar = st.progress(0)  # Initialize progress bar

    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img_np = np.array(img)

            # Lenient thresholding to detect dark spots
            img_np[img_np > threshold_value] = 255  # Set light pixels to white
            img_np[img_np <= threshold_value] = 0   # Set dark pixels to black

            # Find coordinates of dark spots (black pixels)
            y_coords, x_coords = np.where(img_np == 0)  # Find black pixels
            dark_spots.extend(zip(x_coords, y_coords))

            # Create an image with circles around dark spots
            img_processed = Image.fromarray(img_np)
            draw = ImageDraw.Draw(img_processed)
            for (x, y) in dark_spots:
                # Draw a circle around each dark spot
                radius = 8  # Adjust circle radius as needed
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=255, width=2)

            processed_images.append(img_processed)
            
            # Update progress bar
            progress_bar.progress((i + 1) / total_images)

        except Exception as e:
            st.error(f"Error processing image {img_path}: {e}")

    return processed_images, dark_spots

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

# Initialize a global variable to store image files and processed images
if 'image_files' not in st.session_state:
    st.session_state.image_files = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'dark_spots' not in st.session_state:
    st.session_state.dark_spots = []
if 'current_image_index' not in st.session_state:
    st.session_state.current_image_index = 0

# Define the TAR file URL directly
tar_url = "https://github.com/madhan-phy/ML/raw/a7c33130d06525558d75dc1da011372d82daaaad/solar-images/solar_pics.tar.gz"

# Slider for the number of images to fetch
num_images = st.slider("Select number of images to process:", 1, 1000, 500)

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
        st.write("Processing images... Please wait.")
        
        # Process images
        st.session_state.processed_images, st.session_state.dark_spots = process_images(st.session_state.image_files, threshold_value)
        st.success("Image processing complete.")
        st.session_state.current_image_index = 0  # Reset to the first image

# Navigation buttons for image index
if st.session_state.processed_images:
    if st.button("Previous Image"):
        if st.session_state.current_image_index > 0:
            st.session_state.current_image_index -= 1

    if st.button("Next Image"):
        if st.session_state.current_image_index < len(st.session_state.processed_images) - 1:
            st.session_state.current_image_index += 1

    # Display processed image
    current_image = st.session_state.processed_images[st.session_state.current_image_index]
    st.image(current_image, caption=f'Processed Image {st.session_state.current_image_index + 1} of {len(st.session_state.processed_images)}', use_column_width=True)

else:
    st.error("Please fetch images from GitHub and process them first.")

# Option to clean up the folder
if st.button("Clean up image folder"):
    cleanup_temp_folder()
    st.success("Cleaned up the solar_images folder.")
