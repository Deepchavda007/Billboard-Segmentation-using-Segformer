import os
import cv2
import uuid
import shutil
import numpy as np
import tensorflow as tf
from transformers import TFSegformerForSemanticSegmentation
from perspective_transformation import apply_perspective_transformation
from utils import create_response, upload_to_S3, download_file, log_exception


###--------------------------------------------------------------------------###


model_checkpoint = "nvidia/mit-b1"
id2label = {0: "unknown", 1: "billboard"}
label2id = {label: id for id, label in id2label.items()}
num_labels = len(id2label)

# Load the pre-trained model
MODEL = TFSegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
# Model Link : https://drive.google.com/file/d/11-8OLVJ6j6kDQrBL6NY2dYBs6XC7BcWB/view?usp=drive_link
MODEL.load_weights("model/segformer.h5")


###--------------------------------------------------------------------------###


def save_mask_image(predicted_mask, original_image_size, mask_image_path):

    mask = tf.reduce_max(predicted_mask, axis=0)

    # Threshold the mask to create a binary mask
    mask = tf.where(mask >= 0.5, 255, 0)

    _, single_mask = mask
    single_mask_np = np.array(single_mask)

    # Resize the mask to the original image size
    resized_mask = cv2.resize(
        single_mask_np, original_image_size[::-1], interpolation=cv2.INTER_NEAREST
    )

    # Save the mask image to a file
    cv2.imwrite(mask_image_path, resized_mask)


###--------------------------------------------------------------------------###


def process_and_predict_image(image_path, output_dir, batch_id):
    mask_image_path = os.path.join(output_dir, f"{batch_id}_mask.png")

    # Set constants
    image_size = 512
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])

    # Read and decode the image

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    original_image_size = image.shape[:2]

    # Convert image to float32 and resize
    input_image = tf.image.convert_image_dtype(image, tf.float32)
    input_image = tf.image.resize(input_image, (image_size, image_size))

    # Normalize the image
    input_image = (input_image - mean) / tf.maximum(std, tf.keras.backend.epsilon())

    # Transpose image dimensions for the model
    input_image = tf.transpose(input_image, (2, 0, 1))

    def generator_valid():
        # Generator function to yield input images
        yield tf.transpose(input_image, (1, 2, 0))

    test_dataset = tf.data.Dataset.from_generator(
        generator_valid,
        output_signature=tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
    )

    test_dataset = test_dataset.batch(1)
    test_dataset = test_dataset.prefetch(
        tf.data.AUTOTUNE
    )  # Adjust the threshold as needed

    for sample in test_dataset.take(1):
        # Get the images from the dataset
        images = tf.transpose(sample, (0, 3, 1, 2))

        # Make predictions using the model
        pred_masks = MODEL.predict(images).logits

        # Save the predicted masks
        save_mask_image(pred_masks, original_image_size, mask_image_path)

    return mask_image_path


###--------------------------------------------------------------------------###


def process_and_transform_image(payload):

    required_params = ["original_image_url", "replacement_image_url"]
    if any(param not in payload for param in required_params):
        return create_response(False, "Required parameters missing", {}, 400)

    batch_id = payload.get("batch_id", str(uuid.uuid4()))
    output_dir = os.path.join("temp", batch_id)
    os.makedirs(output_dir, exist_ok=True)

    original_image_url = payload.get("original_image_url")
    replacement_image_url = payload.get("replacement_image_url")

    try:
        image_path = download_file(original_image_url, output_dir)
        replacement_image_path = download_file(replacement_image_url, output_dir)

        mask_image_path = process_and_predict_image(image_path, output_dir, batch_id)
        transformed_image_path = apply_perspective_transformation(
            mask_image_path, image_path, replacement_image_path, output_dir, batch_id
        )

        transformed_image_url = upload_to_S3(
            transformed_image_path, f"{batch_id}_transformed_image.png", "output"
        )

        return create_response(
            True,
            "Image processed and transformed successfully",
            {"url": transformed_image_url},
            200,
        )

    except Exception as e:
        log_exception(e)
        return create_response(False, "An error occurred", {}, 500)
    finally:
        shutil.rmtree(output_dir)
