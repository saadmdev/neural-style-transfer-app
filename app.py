import streamlit as st
from PIL import Image
import numpy as np
import os
import io
from style_transfer import perform_style_transfer

st.set_page_config(page_title="🎨 Neural Style Transfer", layout="centered")

st.title("🎨 Neural Style Transfer App")
st.markdown("Upload a content image and a style image to blend them using AI.")

# Upload images
content_image = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_image = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

if content_image and style_image:
    # Save uploads
    os.makedirs("images", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    content_path = os.path.join("images", content_image.name)
    style_path = os.path.join("images", style_image.name)

    with open(content_path, "wb") as f:
        f.write(content_image.read())
    with open(style_path, "wb") as f:
        f.write(style_image.read())

    # Show uploaded images
    st.subheader("📷 Input Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(content_path, caption="Content Image", width=300)
    with col2:
        st.image(style_path, caption="Style Image", width=300)

    if st.button("✨ Generate Stylized Image"):
        st.info("Generating... please wait ⏳")

        output_image = perform_style_transfer(content_path, style_path)

        # Convert NumPy array to PIL image
        styled_image = Image.fromarray(output_image)

        # Show the image in Streamlit
        st.image(styled_image, caption="🎨 Styled Image", use_column_width=True)

        # Save the image
        output_path = os.path.join("output", "stylized_output.png")
        styled_image.save(output_path)

        # Download button (file version)
        with open(output_path, "rb") as file:
            st.download_button("⬇️ Download Result", data=file, file_name="styled_image.png", mime="image/png")

        # Download button (memory buffer)
        buffer = io.BytesIO()
        styled_image.save(buffer, format="PNG")
        st.download_button(
            label="💾 Download Styled Image (Buffer)",
            data=buffer.getvalue(),
            file_name="styled_image.png",
            mime="image/png"
        )

        st.success("Done! 🎉")
