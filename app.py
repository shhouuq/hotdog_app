import streamlit as st
from transformers import pipeline
from PIL import Image

pipe = pipeline("image-classification", model="julien-c/hotdog-not-hotdog")

upload_file = st.file_uploader("Choose a file")


if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img)
    predictions = pipe(img)

    st.write(predictions)
