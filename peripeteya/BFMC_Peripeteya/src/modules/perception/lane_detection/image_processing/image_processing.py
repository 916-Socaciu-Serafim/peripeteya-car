import os
from copy import deepcopy

import cv2
import numpy as np
import glob
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from src.modules.perception.lane_detection.video_processing.video_processing import read_video_to_frames


def test_image():
    test_img_path = "../test_images/notnow/image_c0_b40.jpg"
    img = cv2.imread(test_img_path)
    img = proper_size(img)
    return img


def init_tensors_image():
    path = "init_tensors.jpg"
    img = cv2.imread(path)
    img = proper_size(img)
    return img


def load_test_images():
    path = "../test_images/notnow/*"
    images = []
    for sub_path in glob.glob(path):
        try:
            img = cv2.imread(sub_path)
            images.append(img)
        except Exception:
            continue
    return images


def grayscale(image: np.ndarray):
    """
    Transform an image to grayscale (from 3 layers RGB to one monochromatic layer 0-255 values)
    :param image: numpy.ndarray
    :return: numpy.ndarray
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binary_threshold(image: np.ndarray, threshold=90):
    """
    Filter an image to have only 2 possible values in the matrix 0 (black) or 255 (white)
    :param threshold: int
    :param image: np.ndarray
    :return: np.ndarray (threshold image result)
    """
    return cv2.threshold(image, threshold, 255, type=cv2.THRESH_BINARY)[1]


def mean_threshold_adaptive(image: np.ndarray):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


def gaussian_threshold_adaptive(image: np.ndarray):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


def image_blur(image: np.ndarray, blur_size = (3, 3)):
    """
    Blur an image
    :param image: np.ndarray
    :return: np.ndarray (blurred image)
    """
    return cv2.blur(image, blur_size)


def canny_edge_detection(image: np.ndarray):
    """
    Apply canny edge detection to an image
    :param image: np.ndarray
    :return: np.ndarray (Edges detected by canny function from cv2)
    """
    return cv2.Canny(image, 180, 180, 100, 3)
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
    dest_rect = np.float32([[100, 0], [width - 100, 0], [100, height], [width - 100, height]])
    perspective_matrix = cv2.getPerspectiveTransform(source_rect, dest_rect)
    return cv2.warpPerspective(image, perspective_matrix, dsize=(width, height))


def fill_image(image, horizontal_fill, vertical_fill, color):
    background = np.full((image.shape[0] + vertical_fill * 2, image.shape[1] + horizontal_fill * 2, image.shape[2]),
                         color, dtype=np.uint8)
    background[vertical_fill:vertical_fill + image.shape[0], horizontal_fill:horizontal_fill + image.shape[1]] = image
    return background


def skeleton(image):
    image = image
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, image = cv2.threshold(image, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True
    return skel


def proper_size(image):
    img = cv2.resize(image, (480, 270))
    img = img[img.shape[1] // 5:]
    return img


def generate_images(img, padding=170):
    results = {}
    # images results
    img_grayscale = grayscale(img)
    img_threshold = binary_threshold(img_grayscale)
    img_blur = image_blur(img_threshold * 3)
    img_canny = canny_edge_detection(img)
    warped = perspective_transform(img_canny, *[[padding, 0], [-padding, 0], [0, 0], [0, 0]])
    warped = binary_threshold(warped)
    warped_thresh = perspective_transform(img_threshold, *[[padding, 0], [-padding, 0], [0, 0], [0, 0]])
    warped_canny = perspective_transform(img_canny, *[[padding, 0], [-padding, 0], [0, 0], [0, 0]])
    multi_thresh = multi_threshold(img_grayscale)
    # store in dictionary
    results["grayscale"] = img_grayscale
    # results["threshold"] = img_threshold * 3
    # results["blurred"] = img_blur
    results["canny"] = img_canny
    # results["warped"] = warped
    results["warped_thresh"] = warped_thresh
    results["warped_canny"] = warped_canny
    results["global_thresh"] = multi_thresh["global"]
    results["otsu_tresh"] = multi_thresh["otsu"]
    results["otsu_gaussian"] = multi_thresh["gaussian_otsu"]

    return results


def annotate_results(results):
    # annotate each image
    for result in results:
        results[result] = cv2.putText(results[result], result, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    return results


def annotate_image(image: np.ndarray, text: str, offset: tuple, color=(255, 255, 255)):
    cv2.putText(image, text, offset, cv2.FONT_HERSHEY_COMPLEX, 0.5, color)


def display_images(results):
    # concatenate the images
    for img in results:
        results[img] = cv2.resize(results[img], (240, 120))
    full_result = np.concatenate(list(results.values()), axis=0)
    cv2.imshow("full", full_result)


def hough_lines(image: np.ndarray):
    return cv2.HoughLinesP(image, 1, np.pi / 180, 50, None, 50, 10)


def white_points_histogram(image: np.ndarray, tolerance=20):
    histogram = []
    width = image.shape[1]
    for x in range(width):
        column = image[:, x]
        white_count = np.count_nonzero(column)
        histogram.append(white_count if white_count > tolerance else 0)
    return histogram


def accumulation_points_list(histogram: list, size: int):
    """
    Get maximum number of appearances in a histogram
    :param histogram: list of integers
    :param n: number of maximal points to choose
    :return: maximal points x coordinate
    """
    accumulators = []
    for x in range(len(histogram)):
        value = 0
        for i in range(x-size, x+size):
            try:
                value += histogram[i]
            except Exception:
                continue
        accumulators.append(value)
    return accumulators

def max_points(list):
    return find_peaks(list)[0]
    pass


def multi_threshold(image: np.ndarray):
    # global thresholding
    ret1, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {
        "global": th1,
        "otsu": th2,
        "gaussian_otsu": th3
    }


def lane_start_points(warped_binary: np.ndarray):
    warped = warped_binary
    his = white_points_histogram(warped, 12)
    his = accumulation_points_list(his, 10)
    his = accumulation_points_list(his, 5)
    his = accumulation_points_list(his, 5)

    peaks = max_points(his)
    return peaks


def split_horizontally(image: np.ndarray, n: int):
    # split an image into smaller images horizontally
    # n = number of resulting slices
    image_height = image.shape[0]
    slice_height = image_height//n
    slices = []
    for split in range(n):
        slice_part = image[split * slice_height: (split+1) * slice_height]
        slices.append(slice_part)
    return slices


def blur_thresh(image, threshold, iterations):
    for i in range(iterations):
        image = image_blur(image)
        image = binary_threshold(image, threshold)
    return image


# distances = []
# fav = [0., 0., 0.]
# total = 0.
# images = read_video_to_frames("../image_processing/video_test.mp4")

# for img in images:
#     img = proper_size(img)
#     results = generate_images(img)
#     blur = image_blur(results["warped_canny"])
#     blur = binary_threshold(blur, 4)
#     blur = image_blur(blur)
#     blur = binary_threshold(blur, 4)
#
#     slices = split_horizontally(results["warped_thresh"], 3)
#     for i in range(len(slices)):
#         # cv2.imshow(f"slice{i}", slices[i])
#         start_points = lane_start_points(slices[i])
#         if len(start_points) > 1:
#             distance = start_points[1] - start_points[0]
#             if 170 < distance < 230:
#                 distances.append(distance)
#                 fav[i] += 1.
#         elif len(start_points) == 1:
#             distances.append(210)
#             fav[i] += 1
#     total += 1.
#     # cv2.waitKey(10)
#
#
# print(max(distances), min(distances), np.mean(np.asarray(distances)))
# for i in range(3):
#     print(f"Slice {i}", fav[i]/total)
