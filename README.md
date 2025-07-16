
# Neural Style Transfer Web Application

This project is a web application built with Streamlit that performs neural style transfer, allowing users to blend a content image with a style image using a deep learning model (VGG19). The result is a stylized image that reflects the content of the first image and the artistic style of the second.

### Live Demo  
[![View Live](https://img.shields.io/badge/View%20Live-%231DA1F2?style=for-the-badge&logo=streamlit&logoColor=white)](https://neural-style-transfer-app-rzrbpjunx3wrlaeiyytrgm.streamlit.app)

## Features

- Upload and preview both content and style images through a user interface.
- Performs style transfer using a pre-trained convolutional neural network (VGG19).
- Allows users to download the generated stylized image.
- Clean UI with progress indicator for processing.

## Technologies Used

- Python 3.10.x
- TensorFlow and Keras
- Streamlit
- NumPy
- Pillow (PIL)

## Project Structure

```

neural-style-transfer-app/
│
├── app.py                  # Streamlit frontend interface
├── style\_transfer.py       # Core neural style transfer logic
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
│
├── images/                 # Uploaded content and style images
└── output/                 # Generated stylized output images

````

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/saadmdev/neural-style-transfer-app.git
cd neural-style-transfer-app
````

### 2. Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## How It Works

The backend uses a convolutional neural network (VGG19) to extract features from the content and style images. These features are used to compute content and style losses. The generated image is optimized to minimize these losses, producing a stylized result.

The frontend is built with Streamlit. It allows users to upload images, view the results, and download the final stylized output.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Author

Developed by **Muhammad Saad**
