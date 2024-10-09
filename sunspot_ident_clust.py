import streamlit as st
import numpy as np
from PIL import Image
import requests
import tarfile
import os
import platform
import io
import random
import shutil
def install_libgl():
    system = platform.system()
    
    if system == "Linux":
        # For Debian/Ubuntu
        os.system('sudo apt-get install -y libgl1-mesa-glx')
    elif system == "RedHat":
        # For Red Hat/CentOS
        os.system('sudo yum install -y mesa-libGL')
    elif system == "Darwin":
        # For macOS
        os.system('brew install glfw')  # Assumes Homebrew is installed
    else:
        print("Unsupported OS for dynamic installation.")

# Call the function
install_libgl()
os.system('pip install opencv-python numpy')
import cv2



# Function to find dark spots using OpenCV
def find_dark_spots(image, threshold_value):
    img_np = np.array(image)  # Convert PIL image to NumPy array
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    _, img_binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)  # Invert the thresholding

    # Find contours of dark spots
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_spots = []

    # Loop through contours to get the centroids of dark spots
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small areas
            M = cv2.moments(contour)
            if M['m00'] != 0:  # Avoid division by zero
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                dark_spots.append((cX, cY))

    return dark_spots

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
    img_with_circles = np.array(st.session_state.original_image)
    
    # Draw circles around detected dark spots
    for (x, y) in st.session_state.dark_spots:
        cv2.circle(img_with_circles, (x, y), 5, (255, 0, 0), 2)  # Draw a small circle around the spot

    st.image(img_with_circles, caption="Original Image with Dark Spots", use_column_width=True)

# Option to clean up the folder
if st.button("Clean up image folder"):
    cleanup_temp_folder()
    st.success("Cleaned up the solar_images folder.")
