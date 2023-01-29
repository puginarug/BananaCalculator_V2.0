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

st.set_page_config(page_title="Is My Banana Ripe?",page_icon=":banana:",layout="wide")

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

