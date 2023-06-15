import cv2
import numpy as np
import urllib.request
import pandas as pd
from urllib.error import HTTPError
import streamlit as st
st.set_page_config(layout="wide")

threshold = 80

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    urls = data.iloc[:, 0].tolist()  # assuming urls are in the first column of the CSV file
else:
    urls = []  # fallback to an empty list if no file is uploaded

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0)
my_text = st.empty()

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'}

output = []

for i, url in enumerate(urls):
    current_progress_text = f"Processing URL {i + 1}/{len(urls)}: {url}"
    my_text.text(current_progress_text)
    my_bar.progress((i + 1) / len(urls))

    try:
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req)

        image_size = response.headers['Content-Length']  # Get the image size from the headers

        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        n_white_pix = np.sum(np.all(img == [255, 255, 255], axis=-1))
        total_pix = img.shape[0] * img.shape[1]  # Total number of pixels in the image
        white_pix_percentage = (n_white_pix / total_pix) * 100  # Percentage of white pixels

        wbi = 1 if white_pix_percentage > threshold else 0
        the_result = {"url": url, "image_link": url, "white_px_count": n_white_pix, "wbi": wbi, "error": "", "image_size": image_size, "white_pix_percentage": white_pix_percentage}
        output.append(the_result)
    except HTTPError as e:
        error_message = f"HTTPError occurred for URL: {url}. Error code: {e.code}"
        print(error_message)
        the_result = {"url": url, "image_link": "", "white_px_count": 0, "wbi": 0, "error": error_message, "image_size": "N/A", "white_pix_percentage": "N/A"}
        output.append(the_result)

df = pd.DataFrame(output, columns=['url', 'image_link', 'white_px_count', 'wbi', 'error', 'image_size', 'white_pix_percentage'])

st.data_editor(
    df,
    column_config={
        "url": st.column_config.ImageColumn(
            "Preview Image", help="Streamlit app preview screenshots"
        )
    },
    hide_index=True,
)

df.to_csv('wbi.csv')
