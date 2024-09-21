import os
import cv2
import numpy as np


###--------------------------------------------------------------------------###


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


###--------------------------------------------------------------------------###


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped, rect, dst, maxWidth, maxHeight


###--------------------------------------------------------------------------###


def four_point_transform_inverse(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[maxWidth - 1, 0], [0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1]],
        dtype="float32",
    )
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(dst, rect)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped, rect, dst, maxWidth, maxHeight


###--------------------------------------------------------------------------###


def find_contour_mask(mask_image_path):

    # Read the mask image and convert it to grayscale
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the mask image
    blurred = cv2.GaussianBlur(mask_image, (3, 3), 0)

    # Apply Otsu's thresholding to the blurred image to create a binary image
    _, thresholded_image = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Perform morphological opening and dilation to get the background region
    sure_bg = cv2.dilate(
        cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2),
        kernel,
        iterations=3,
    )

    # Find contours in the sure_bg image
    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


###--------------------------------------------------------------------------###


def apply_perspective_transformation(
    mask_image_path, original_image_path, advertisement_path, output_dir, batch_id
):
    """
    Apply perspective transformation to replace a masked region in the original image
    with an advertisement image, based on the given mask.

    Args:
        mask_image_path (str): Path to the mask image.
        original_image_path (str): Path to the original image.
        advertisement_path (str): Path to the advertisement image.
        output_dir (str): Directory to save the output image.
        batch_id (str): Batch identifier for naming the output file.

    Returns:
        str: Path to the saved image after applying perspective transformation.
    """

    # Find contours in the mask image
    contours = find_contour_mask(mask_image_path)
    transformed_image_path = os.path.join(output_dir, f"{batch_id}_advertisement.png")

    # Create a color version of the mask image for drawing lines
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    replacement_image = cv2.imread(advertisement_path)
    original_image = cv2.imread(original_image_path)

    mask_with_lines = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    # Iterate over each contour
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Filter out small contours
        if area > 1000:
            # Approximate the contour with a polygon
            epsilon = 0.07 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Draw the polygon on the mask_with_lines image
            cv2.polylines(
                mask_with_lines, [approx], isClosed=True, color=(0, 255, 0), thickness=2
            )

            # Reshape the contour to get the corners of the polygon
            corners = approx.reshape(-1, 2)

            # Perform perspective transformation on the mask image using the corners
            transformed_mask_image, src_pts, dst_pts, max_Width, max_Height = (
                four_point_transform(mask_image, corners)
            )

            # Perform perspective transformation on the original image using the corners
            transformed_original_image, src_pts, dst_pts, max_Width, max_Height = (
                four_point_transform(original_image, corners)
            )

            # Resize the replacement image to match the transformed original image dimensions
            replacement_image_resized = cv2.resize(
                replacement_image,
                (
                    transformed_original_image.shape[1],
                    transformed_original_image.shape[0],
                ),
            )

            # Overlay the resized replacement image on the transformed original image
            transformed_original_image[
                0 : replacement_image_resized.shape[0],
                0 : replacement_image_resized.shape[1],
            ] = replacement_image_resized

            # Convert the transformed original image to RGB format
            transformed_original_image_rgb = cv2.cvtColor(
                transformed_original_image, cv2.COLOR_BGR2RGB
            )

            # Obtain the perspective transformation matrix from destination to source points
            M = cv2.getPerspectiveTransform(dst_pts, src_pts)

            # Warp the transformed original image back to the original perspective
            warped = cv2.warpPerspective(
                transformed_original_image_rgb,
                M,
                (original_image.shape[1], original_image.shape[0]),
            )

            # Convert original image and result image to RGB format
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            result_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

            # Create a mask of black pixels in the result image
            result_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(result_gray, 1, 255, cv2.THRESH_BINARY)[1]

            # Invert the mask to select black pixels
            inv_mask = cv2.bitwise_not(mask)

            # Apply the mask to the result image
            masked_result = cv2.bitwise_and(warped, warped, mask=mask)

            # Apply the inverted mask to the original image
            masked_original = cv2.bitwise_and(original_rgb, original_rgb, mask=inv_mask)

            # Combine the masked images to get the final result
            final_image = cv2.add(masked_result, masked_original)

            # Convert the final image to RGB format
            final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

            cv2.imwrite(transformed_image_path, final_image_rgb)

    return transformed_image_path
