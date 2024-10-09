import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import requests
import tarfile
import os
import io
import random  # Import random for image selection
from sklearn.cluster import KMeans
import shutil

# Function to process images and mark sunspots
def process_images(image_files, threshold_value):
    processed_images = []
    sunspots = []
    
    total_images = len(image_files)
    progress_bar = st.progress(0)  # Initialize progress bar

    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img_np = np.array(img)

            # Simple thresholding to detect sunspots (black dots)
            img_np[img_np > threshold_value] = 255
            img_np[img_np <= threshold_value] = 0

            # Create an edge-detected version
            edges = np.zeros_like(img_np)
            for y in range(1, img_np.shape[0] - 1):
                for x in range(1, img_np.shape[1] - 1):
                    # Simple Sobel-like edge detection
                    gx = img_np[y-1, x+1] - img_np[y-1, x-1] + \
                         2 * (img_np[y, x+1] - img_np[y, x-1]) + \
                         img_np[y+1, x+1] - img_np[y+1, x-1]

                    gy = img_np[y+1, x-1] - img_np[y-1, x-1] + \
                         2 * (img_np[y+1, x] - img_np[y-1, x]) + \
                         img_np[y+1, x+1] - img_np[y-1, x+1]

                    edge_magnitude = np.sqrt(gx**2 + gy**2)
                    edges[y, x] = 255 if edge_magnitude > 50 else 0  # Threshold to define edges

            # Create a mask to exclude detected lines
            mask = np.ones_like(img_np, dtype=bool)

            # Check for potential lines in the edge-detected image
            for y in range(edges.shape[0]):
                for x in range(edges.shape[1]):
                    if edges[y, x] == 255:  # Found an edge
                        # Mark the surrounding pixels (adjust if needed)
                        mask[max(0, y-1):min(y+2, mask.shape[0]), max(0, x-1):min(x+2, mask.shape[1])] = False

            # Exclude pixels near the detected lines
            img_np[~mask] = 255  # Set masked areas to white

            # Find remaining sunspots
            y_coords, x_coords = np.where(img_np == 0)  # Find remaining black pixels
            sunspots.extend(zip(x_coords, y_coords))

            img_processed = Image.fromarray(img_np)
            for (x, y) in sunspots:
                img_processed.putpixel((x, y), 255)  # Change sunspot pixels to white

            processed_images.append(img_processed)
            
            # Update progress bar
            progress_bar.progress((i + 1) / total_images)

        except Exception as e:
            st.error(f"Error processing image {img_path}: {e}")

    return processed_images, sunspots

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
st.title("Solar Sunspot Analysis from Images")

# Initialize a global variable to store image files and processed images
if 'image_files' not in st.session_state:
    st.session_state.image_files = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'sunspots' not in st.session_state:
    st.session_state.sunspots = []

# Define the TAR file URL directly
tar_url = "https://github.com/madhan-phy/ML/raw/a7c33130d06525558d75dc1da011372d82daaaad/solar-images/solar_pics.tar.gz"

# Slider for the number of images to fetch
num_images = st.slider("Select number of images to process:", 10, 1000, 500)

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
threshold_value = st.slider("Select Threshold Value for Sunspot Detection:", 0, 255, 50)

if st.button("Process Images"):
    if st.session_state.image_files:  # Check if image_files is not empty
        st.write("Processing images... Please wait.")
        
        # Process images
        st.session_state.processed_images, st.session_state.sunspots = process_images(st.session_state.image_files, threshold_value)
        st.success("Image processing complete.")
        
        # Initialize index for slider
        st.session_state.current_image_index = 0

# Slider for image index
if st.session_state.processed_images:
    image_index = st.slider("Select an image index:", 0, len(st.session_state.processed_images) - 1, 0)
    
    # Display processed image
    st.image(st.session_state.processed_images[image_index], caption=f'Processed Image {image_index + 1} of {len(st.session_state.processed_images)}', use_column_width=True)

    # Display sunspot information
    st.write("### Detected Sunspots:")
    for idx, (x, y) in enumerate(st.session_state.sunspots):
        st.write(f"Sunspot {idx + 1}: Coordinates (X: {x}, Y: {y})")

    # If clustering is enabled, apply K-Means
    if st.session_state.sunspots:
        sunspot_coords = np.array(st.session_state.sunspots)
        num_clusters = st.slider("Select number of clusters:", 1, 10, 3)
        
        with st.spinner("Clustering sunspots..."):
            kmeans = KMeans(n_clusters=num_clusters)
            clusters = kmeans.fit_predict(sunspot_coords)
            st.success("Clustering complete.")

        cluster_df = pd.DataFrame(sunspot_coords, columns=['X', 'Y'])
        cluster_df['Cluster'] = clusters

        fig_cluster = px.scatter(cluster_df, x='X', y='Y', color='Cluster',
                                  title='Sunspot Clusters',
                                  labels={'X': 'X Coordinate', 'Y': 'Y Coordinate'})
        st.plotly_chart(fig_cluster)
else:
    st.error("Please fetch images from GitHub and process them first.")

# Option to clean up the folder
if st.button("Clean up image folder"):
    cleanup_temp_folder()
    st.success("Cleaned up the solar_images folder.")
