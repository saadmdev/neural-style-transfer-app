# Neural Style Transfer Web Application

This project is a web application built with Streamlit that performs neural style transfer, allowing users to blend a content image with a style image using a deep learning model (VGG19). The result is a stylized image that reflects the content of the first image and the artistic style of the second.

## Features

- Upload and preview both content and style images through a user interface.
- Performs style transfer using a pre-trained convolutional neural network (VGG19).
- Allows users to download the generated stylized image.
- Clean UI with progress indicator for processing.

## Technologies Used

- Python 3.x
- TensorFlow and Keras
- Streamlit
- NumPy
- Pillow (PIL)

## Project Structure

neural-style-transfer-app/
│
├── app.py # Streamlit frontend
├── style_transfer.py # Backend logic for style transfer
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── images/ # Uploaded content/style images
└── output/ # Saved stylized images

bash
Copy
Edit

## Installation & Setup

1. **Clone the repository:**

```bash
git clone https://github.com/saadmdev/neural-style-transfer-app.git
cd neural-style-transfer-app
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv venv
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate
Install the dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
How It Works
The backend uses a convolutional neural network (VGG19) to extract features from the content and style images. The model then generates an output image that minimizes the content loss and style loss, using gradient descent optimization.

The app interface handles image uploads, display, and triggering the transfer process. After processing, the output image can be viewed and downloaded.

License
This project is open source and available under the MIT License.

Author
Developed by Muhammad Saad

