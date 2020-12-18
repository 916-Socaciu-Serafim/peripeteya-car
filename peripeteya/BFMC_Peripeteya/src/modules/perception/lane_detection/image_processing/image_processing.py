import os

import cv2
import numpy as np
import glob


def test_image():
    test_img_path = "../test_images/test19.png"
    img = cv2.imread(test_img_path)
    img = cv2.resize(img, (240, 240))[100:]
    return img


def load_test_images():
    path = "../test_images/*"
    images = []
    for sub_path in glob.glob(path):
        try:
            img = cv2.imread(sub_path)
            img = cv2.resize(img, (240, 240))[100:]
            images.append(img)
        except Exception:
            continue
    print(len(images))
    return images


def grayscale(image: np.ndarray):
    """
    Transform an image to grayscale (from 3 layers RGB to one monochromatic layer 0-255 values)
    :param image: numpy.ndarray
    :return: numpy.ndarray
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binary_threshold(image: np.ndarray):
    """
    Filter an image to have only 2 possible values in the matrix 0 (black) or 255 (white)
    :param image: np.ndarray
    :return: np.ndarray (threshold image result)
    """
    return cv2.threshold(image, 100, 255, type=cv2.THRESH_BINARY)[1]


def image_blur(image: np.ndarray):
    """
    Blur an image
    :param image: np.ndarray
    :return: np.ndarray (blurred image)
    """
    return cv2.blur(image, (3, 3))


def canny_edge_detection(image: np.ndarray):
    """
    Apply canny edge detection to an image
    :param image: np.ndarray
    :return: np.ndarray (Edges detected by canny function from cv2)
    """
    return cv2.Canny(image, 100, 200)
    pass


def perspective_transform(image, *offsets):
    """
    Warp the perspective of an image given the rectangle points (destination size is same as image.shape)
    :param image: np.ndarray
    :return: warped image
    """
    pos1 = offsets[0]
    pos2 = offsets[1]
    pos3 = offsets[2]
    pos4 = offsets[3]
    height = image.shape[0]
    width = image.shape[1]
    source_rect = np.float32([pos1,
                              [width + pos2[0], pos2[1]],
                              [pos3[0], height + pos3[1]],
                              [width + pos4[0], height + pos4[1]]
                              ])
    dest_rect = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    perspective_matrix = cv2.getPerspectiveTransform(source_rect, dest_rect)
    return cv2.warpPerspective(image, perspective_matrix, dsize=(image.shape[1], image.shape[0]))


def fill_image(image, horizontal_fill, vertical_fill, color):
    background = np.full((image.shape[0] + vertical_fill * 2, image.shape[1] + horizontal_fill * 2, image.shape[2]),
                         color, dtype=np.uint8)
    background[vertical_fill:vertical_fill + image.shape[0], horizontal_fill:horizontal_fill + image.shape[1]] = image
    return background


def generate_images(img, padding=110):
    results = {}
    # images results
    img = cv2.resize(img, (240, 240))[100:]
    img = fill_image(img, padding, 0, (0, 0, 0))
    img_grayscale = grayscale(img)
    img_threshold = binary_threshold(img_grayscale)
    img_blur = image_blur(img_threshold * 3)
    img_canny = canny_edge_detection(img_blur)
    warped = perspective_transform(img_grayscale, *[[padding, 0], [-padding, 0], [0, 0], [0, 0]])
    warped = binary_threshold(warped)
    # store in dictionary
    results["grayscale"] = img_grayscale
    results["threshold"] = img_threshold * 3
    results["blurred"] = img_blur
    results["canny"] = img_canny
    results["warped"] = warped
    return results


def annotate_results(results):
    # annotate each image
    for result in results:
        results[result] = cv2.putText(results[result], result, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    return results


def display_images(results):
    # concatenate the images
    full_result = np.concatenate(list(results.values()), axis=0)
    cv2.imshow("full", full_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


