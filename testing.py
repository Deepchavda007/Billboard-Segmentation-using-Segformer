import tensorflow as tf
import numpy as np
from transformers import TFSegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import cv2
import os

class BillboardSegmentationModel:
    def __init__(self):
        # Initialize the model checkpoint, label mappings, and the model itself
        self.model_checkpoint = "mit-b5"
        self.id2label = {0: "unknown", 1: "billboard"}
        self.label2id = {label: id for id, label in self.id2label.items()}
        self.num_labels = len(self.id2label)
        self.model = None
        self.load_model()

    def load_model(self):
        # Load the pre-trained model and its weights
        self.model = TFSegformerForSemanticSegmentation.from_pretrained(
            self.model_checkpoint,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.model.load_weights("Billboard Segmentation 75.h5")
        # clear_output()
        # print("-" * 23, "Model loaded successfully", "-" * 23)

    def save_predicted_mask(self, mask, original_image_size, new_img_name):
        # Assuming `mask` is a tensor representing the predicted mask
        mask = tf.reduce_max(mask, axis=0)  # Convert multichannel mask to a single channel

        # Threshold the mask to create a binary mask
        mask = tf.where(mask >= 0.5, 255, 0)  # Adjust the threshold as needed

        i, single_mask = mask
        # Convert the mask to a NumPy array
        single_mask_np = np.array(single_mask)

        # Resize the mask to the original image size
        single_mask_resized = cv2.resize(single_mask_np, original_image_size[::-1], interpolation=cv2.INTER_NEAREST)

        # Display the resized mask
        plt.imshow(single_mask_resized)
        plt.show()

        # Save the mask image to a file
        cv2.imwrite(new_img_name, single_mask_resized)

    def create_and_show_predictions(self, image_path):
        """
        Reads an image from the given path, preprocesses it, and makes predictions using a model.
        The predicted masks are then saved.

        Args:
            image_path (str): The path to the input image.

        Returns:
            str: A success message indicating that the predicted masks have been saved.
        """

        # Set constants
        image_size = 512
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])

        # Read and decode the image
        img_name = os.path.basename(image_path).rsplit(".", 1)[0]
        new_img_name = f"{img_name}_mask.png"

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

        # Create a dataset from the generator
        test_dataset = tf.data.Dataset.from_generator(
            generator_valid,
            output_signature=tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32)
        )

        # Batch and prefetch the dataset
        test_dataset = test_dataset.batch(1)
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

        for sample in test_dataset.take(1):
            # Get the images from the dataset
            images = tf.transpose(sample, (0, 3, 1, 2))

            # Make predictions using the model
            pred_masks = self.model.predict(images).logits

            # Save the predicted masks
            self.save_predicted_mask(pred_masks, original_image_size, new_img_name)

        return "Saved Mask_Image Successfully"


if __name__ == "__main__":

    # Instantiate the BillboardSegmentationModel
    model = BillboardSegmentationModel()

    # Define the path to the input image
    image_path = "Test/billboard.png"

    # Create and show predictions
    result = model.create_and_show_predictions(image_path)

    # Print the result
    print(result)