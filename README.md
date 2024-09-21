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
   <h3>This project utilizes the Segformer pretrained model to perform billboard segmentation through semantic segmentation. It specifically targets Indian billboards. By employing transfer learning on a custom dataset, the model precisely delineates billboards within images, allowing for efficient detection and analysis.
</h3>
  <h4>Welcome to the setup guide for the Billboard Segmentation Module. Follow these steps to get your environment ready and run the application.</h4>
</p>

## Project Overview

This project is split into two key sections:
1. **Training**: We fine-tune the Segformer model on a custom Indian billboard dataset using semantic segmentation techniques.
2. **API for Billboard Replacement**: Once the billboards are segmented, the API provides a method to replace billboards using a **perspective transformation** approach to accurately fit new content onto existing billboards.

For details on training the model, follow the dedicated [Training README](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/blob/main/notebook/README.md) and explore the [Google Colab Notebook](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/blob/main/notebook/Billboard_Segmentation.ipynb) for downstream tasks.

## Prerequisites
- Ensure you have Git installed on your system.
- Python 3 should be installed on your system.

## Step 1: Clone the Repository
First, clone the repository to your local machine using Git. Open your terminal and run:

```
git clone https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer.git
```

## Step 2: Create a Virtual Environment
Creating a virtual environment is crucial to manage dependencies.

### For Mac & Linux:
Run the following commands:

```
python3 -m venv env/segmentation
source env/segmentation/bin/activate
```

### For Windows:
Run these commands in your Command Prompt or PowerShell:

```
python -m venv env\segmentation
.\env\segmentation\Scripts\activate
```

With your virtual environment active, install the required Python packages:


#### For Windows:
```
pip install -r requirements.txt
```

#### For Mac & Linux:
```
pip3 install -r requirements.txt
```

### Step 5: Run the Flask API

Run the Flask API using the following command:

```bash
python3 app.py
```

## API Endpoints:
  - The application provides the following endpoints:
### /transform_image
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
## Training Setup

For those interested in training or fine-tuning the Segformer model, follow the dedicated training instructions in the [Training README](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/blob/main/notebook/README.md) or use the provided [Google Colab Notebook](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/blob/main/notebook/Billboard_Segmentation.ipynb) for downstream tasks such as segmentation and billboard replacement.

## File Structure
The directory structure of the codebase is organized as follows:
- `app/` – Contains Flask application files.
- `Data/` – Billboard training and validation datasets.
- `notebook/` – Jupyter notebooks for training and evaluations.
- `requirements.txt` – Lists all dependencies.
  
## Conclusion
Your setup is now complete! If you encounter any issues, submit an issue on the GitHub repository.

## Contribution
<a href="https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Deepchavda007/Billboard-Segmentation-using-Segformer" />
</a>
