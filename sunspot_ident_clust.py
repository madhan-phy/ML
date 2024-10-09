import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import requests
import tarfile
import os
import io
from sklearn.cluster import KMeans
import shutil

# Function to download images from a URL
def download_image(url):
    response = requests.get(url)
    img = Image.open(requests.get(url, stream=True).raw)
    return img

# Function to process images and mark sunspots
def process_images(image_files, threshold_value):
    processed_images = []
    sunspots = []

    for img_path in image_files:
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        img_np = np.array(img)

        # Simple thresholding to detect sunspots (black dots)
        img_np[img_np > threshold_value] = 255
        img_np[img_np <= threshold_value] = 0

        # Convert back to PIL image for display
        img_processed = Image.fromarray(img_np)

        # Find sunspots (coordinates of black pixels)
        y_coords, x_coords = np.where(img_np == 0)  # Black pixels
        sunspots.extend(zip(x_coords, y_coords))

        # Optionally, draw the sunspots on the processed image
        for (x, y) in sunspots:
            img_processed.putpixel((x, y), 255)  # Change black dots to white

        processed_images.append(img_processed)

    return processed_images, sunspots

# Function to download and extract TAR file from GitHub
def download_and_extract_tar(username, repo_name, tar_path):
    url = f"https://github.com/{username}/{repo_name}/archive/refs/heads/{tar_path}.tar.gz"
    response = requests.get(url)

    if response.status_code == 200:
        with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as tar:
            tar.extractall("temp_images")  # Extract to temp_images folder
        return [os.path.join("temp_images", f) for f in os.listdir("temp_images") if f.endswith('.jpg')]
    else:
        st.error("Failed to retrieve TAR file from GitHub.")
        return []

# Function to clean up temp folder
def cleanup_temp_folder():
    if os.path.exists("temp_images"):
        shutil.rmtree("temp_images")

# Streamlit app layout
st.title("Solar Sunspot Analysis from Images")

# Clean up temp folder at the start
cleanup_temp_folder()

# Input: GitHub Repository Info
username = st.text_input("Enter your GitHub username:")
repo_name = st.text_input("Enter your GitHub repository name:")
tar_path = st.text_input("Enter the name of the branch containing the TAR:", "main")  # Default branch

if st.button("Fetch Images from GitHub"):
    image_files = download_and_extract_tar(username, repo_name, tar_path)
    if image_files:
        st.success(f"Fetched {len(image_files)} images from GitHub.")
    else:
        st.error("No images found.")

# Slider for threshold value
threshold_value = st.slider("Select Threshold Value for Sunspot Detection:", 0, 255, 50)

if st.button("Process Images"):
    if 'image_files' in locals() and image_files:
        processed_images, sunspots = process_images(image_files, threshold_value)

        # Slideshow for processed images
        st.subheader("Processed Images with Sunspots Marked")
        image_index = st.slider("Select an image index:", 0, len(processed_images) - 1, 0)
        st.image(processed_images[image_index], caption=f'Processed Image {image_index + 1}', use_column_width=True)

        # Display sunspot information
        st.write("### Detected Sunspots:")
        for idx, (x, y) in enumerate(sunspots):
            st.write(f"Sunspot {idx + 1}: Coordinates (X: {x}, Y: {y})")

        # If clustering is enabled, apply K-Means
        if sunspots:
            sunspot_coords = np.array(sunspots)
            num_clusters = st.slider("Select number of clusters:", 1, 10, 3)
            kmeans = KMeans(n_clusters=num_clusters)
            clusters = kmeans.fit_predict(sunspot_coords)

            cluster_df = pd.DataFrame(sunspot_coords, columns=['X', 'Y'])
            cluster_df['Cluster'] = clusters

            fig_cluster = px.scatter(cluster_df, x='X', y='Y', color='Cluster',
                                      title='Sunspot Clusters',
                                      labels={'X': 'X Coordinate', 'Y': 'Y Coordinate'})
            st.plotly_chart(fig_cluster)
    else:
        st.error("Please fetch images from GitHub before processing.")
