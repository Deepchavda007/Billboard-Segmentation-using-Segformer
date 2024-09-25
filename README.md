<p>
  <div align="center">
  <h1>
<br >
  Billboard Segmentation <br /> <br />
    <a href="https://github.com/psf/black">
      <img
        src="https://img.shields.io/badge/code%20style-black-000000.svg"
        alt="The Uncompromising Code Formatter"
      />
    </a>
      <a>
      <img
        src="https://img.shields.io/badge/python-3.9%20%7C%203.10-blue"
        alt="Python Versions"
      />
    </a>
    <a href="https://docs.opencv.org/4.x/index.htm)">
      <img
        src="https://img.shields.io/badge/CV-OpenCV-red"
        alt="CV-OpenCV-red"
      />
    </a>
    <a href="https://www.tensorflow.org/">
      <img
        src="https://img.shields.io/badge/DL-TensorFlow-orange"
        alt="TensorFlow"
      />
    </a>
    <a href="https://huggingface.co/docs/transformers/model_doc/segformer">
  <img
    src="https://img.shields.io/badge/Semantic%20Segmentation-Segformer-brightgreen"
    alt="Segformer"
  />
</a>
     <a href="https://opensource.org/licenses/MIT">
      <img
        src="https://img.shields.io/badge/License-MIT-blue.svg"
        alt="License: MIT"
      />
    </a>
  </h1>
  </div>
</p>

Use Segformer pretrained model to perform billboard segmentation through semantic segmentation. It specifically targets Indian billboards, and the model precisely delineates billboards within images, allowing for efficient detection and analysis.

## Overview

This project is split into two key sections:

1. **Training**: We fine-tune the Segformer model on a custom Indian billboard dataset using semantic segmentation techniques.
2. **API for Billboard Replacement**: Once the billboards are segmented, the API provides a method to replace billboards using a **perspective transformation** approach to accurately fit new content onto existing billboards.

For details on training the model, follow the dedicated [Training README](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/blob/main/notebook/README.md) and explore the [Google Colab Notebook](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/blob/main/notebook/Billboard_Segmentation.ipynb) for downstream tasks.

### Step 1: Clone the Repository
First, clone the repository to your local machine using Git. Open your terminal and run:

```
git clone https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer.git
```

### Step 2: Create a Virtual Environment
Creating a virtual environment is crucial to manage dependencies.

#### For Mac & Linux:
Run the following commands:

```
python3 -m venv env/segmentation
source env/segmentation/bin/activate
```

#### For Windows:
Run these commands in your Command Prompt or PowerShell:

```
python -m venv env\segmentation
.\env\segmentation\Scripts\activate
```

### Step 3: Install dependencies

With your virtual environment active, install the required Python packages:


#### For Windows:
```
pip install -r requirements.txt
```

#### For Mac & Linux:
```
pip3 install -r requirements.txt
```

### Step 4: Run the Flask API

Run the Flask API using the following command:

```bash
python3 app.py
```

### API Endpoints:
  - The application provides the following endpoints:
#### /transform_image
- **Method**: POST
- **Description**: This endpoint removes the background from a billboard and replaces it with the provided image, using perspective transformation to adjust the replacement image to fit the billboard's orientation and perspective.
- **Request Body**:
  ```json
   {
    "original_image_url": "https://example.com/original.png",
    "replacement_image_url": "https://example.com/replacement.png"
   }
  ```
- **Response**:
  - **Success**:
     ```json
    {
        "data": {
            "url": "https://s3.amazonaws.com/bucketname/path/to/transformed_image.png"
        },
        "message": "Image processed and transformed successfully",
        "status": true
    }
    ```
  
### Training

For those interested in training or fine-tuning the Segformer model, follow the dedicated training instructions in the [Training README](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/blob/main/notebook/README.md) or use the provided [Google Colab Notebook](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/blob/main/notebook/Billboard_Segmentation.ipynb) for downstream tasks such as segmentation and billboard replacement.

## Contribution

<a href="https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Deepchavda007/Billboard-Segmentation-using-Segformer" />
</a>
