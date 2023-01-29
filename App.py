# imports
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
from rembg import remove
from PIL import Image

# -------------------------------------------------------------------------------------
# Page Configuration:

st.set_page_config(page_title="Is My Banana Ripe? V2",page_icon=":banana:",layout="wide")

# -------------------------------------------------------------------------------------
# Interface:

header = st.container()
input = st.container()
output = st.container()

with header:
    st.title("To Eat, Or Not To Eat?")
    st.subheader("Low-Computing-Power Banana Ripening Calculator")
    st.write("This project was built as a part of \"Intro To Image Processing\"\ncourse in the Faculty of Agriculture.\nIt's quite simple:\n")
    st.write("*  Upload an image of a [banana](https://en.wikipedia.org/wiki/Banana) to the \"Input Image\" section.")
    st.write("*  The ripeness status of the banana will be presented in the \"Calculated Status\" section.")

with input:
    st.header("Input Image:")
    loaded_image = st.file_uploader("Upload an image of a banana:", type=['jpg', 'jpeg', 'png'])
    if loaded_image is not None:
        image = Image.open(loaded_image)
        st.image(image, caption='Uploaded banana')
    else:
        st.markdown(f'<h1 style="color:red;font-size:24px;">{"Uploading an image is required"}</h1>', unsafe_allow_html=True)
        
# -------------------------------------------------------------------------------------
# Banana Ripeness Color Scheme:
UNRIPE = np.array([[143, 169, 65]])
SEMI_RIPE = np.array([[213, 209, 122]])
RIPE = np.array([[244, 219, 100]])
HALF_ROTTEN = np.array([[234, 171, 64]])
ROTTEN = np.array([[46, 37, 34]])

# -------------------------------------------------------------------------------------
# steps 1+2: reading the banana image and removing the background using rembg

output_img = remove(image) # remove background

# -------------------------------------------------------------------------------------
# step 3: pre-process the image before segmentation

output_img = np.array(output_img) # convert image to numpy array

output_img = output_img[:,:,:3] # removing the alpha channel (converting from RGBA to RGB)

# -------------------------------------------------------------------------------------
# step 4: K-means segmentation

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = output_img.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(output_img.shape)

# -------------------------------------------------------------------------------------
# step 5: getting the centers of the clusters 

# count how many pixels in each cluster
unique, counts = np.unique(labels, return_counts=True)

# get the rgb values of the centers of the clusters
centers = centers.tolist()

# create a dataframe of the pixel counts and the centers, without the black background
df = pd.DataFrame({'pixel_count': counts, 'centers': centers})

for i in range(len(df)):
    if df['centers'][i] == [0, 0, 0]:
        df.drop(i, inplace=True)
        
# classify the clusters according to the closest color to the RGB color scheme
def classify_cluster(row):

    if np.linalg.norm(row['centers'] - UNRIPE) < np.linalg.norm(row['centers'] - SEMI_RIPE) and np.linalg.norm(row['centers'] - UNRIPE) < np.linalg.norm(row['centers'] - RIPE) and np.linalg.norm(row['centers'] - UNRIPE) < np.linalg.norm(row['centers'] - HALF_ROTTEN) and np.linalg.norm(row['centers'] - UNRIPE) < np.linalg.norm(row['centers'] - ROTTEN):
        return 'Unripe'
    elif np.linalg.norm(row['centers'] - SEMI_RIPE) < np.linalg.norm(row['centers'] - UNRIPE) and np.linalg.norm(row['centers'] - SEMI_RIPE) < np.linalg.norm(row['centers'] - RIPE) and np.linalg.norm(row['centers'] - SEMI_RIPE) < np.linalg.norm(row['centers'] - HALF_ROTTEN) and np.linalg.norm(row['centers'] - SEMI_RIPE) < np.linalg.norm(row['centers'] - ROTTEN):
        return 'Semi-ripe'
    elif np.linalg.norm(row['centers'] - RIPE) < np.linalg.norm(row['centers'] - UNRIPE) and np.linalg.norm(row['centers'] - RIPE) < np.linalg.norm(row['centers'] - SEMI_RIPE) and np.linalg.norm(row['centers'] - RIPE) < np.linalg.norm(row['centers'] - HALF_ROTTEN) and np.linalg.norm(row['centers'] - RIPE) < np.linalg.norm(row['centers'] - ROTTEN):
        return 'Ripe'
    elif np.linalg.norm(row['centers'] - HALF_ROTTEN) < np.linalg.norm(row['centers'] - UNRIPE) and np.linalg.norm(row['centers'] - HALF_ROTTEN) < np.linalg.norm(row['centers'] - SEMI_RIPE) and np.linalg.norm(row['centers'] - HALF_ROTTEN) < np.linalg.norm(row['centers'] - RIPE) and np.linalg.norm(row['centers'] - HALF_ROTTEN) < np.linalg.norm(row['centers'] - ROTTEN):
        return 'Half-Rotten'
    else:
        return 'Rotten'

df['class'] = df.apply(classify_cluster, axis=1)

# calculate the percentage of each class
df['percentage'] = df['pixel_count'] / df['pixel_count'].sum() * 100


# creating a new datafram with everything but thr centers column
df2 = df.drop(columns=['centers'])

# group by ripening level and sum the percentages
df2 = df2.groupby('class').sum()


# deciding the ripening level
# the ripening level is the one with the highest percentage

status = df2['percentage'].idxmax()

# -------------------------------------------------------------------------------------
# Output:

with output:
    st.header(f"Calculated Status: {status}")
    
    # plot the pie chart
    fig1, ax1 = plt.subplots()
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#BB8BEB']
    ax1.pie(y='percentage', figsize=(5, 5), autopct='%1.1f%%', startangle=90, legend=False, fontsize=14, colors=colors)
    ax1.axis
    st.pyplot(fig1)
