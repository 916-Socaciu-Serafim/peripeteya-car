import math
from copy import deepcopy

import cv2
import numpy as np


def lines_detection(image: np.ndarray):
    """
    This one is not working pretty well
    :param image:
    :return:
    """
    copy = deepcopy(image) * 3
    lines = cv2.HoughLinesP(copy, 1, np.pi / 180, 110)
    if lines is not None:
        for line in lines:
            print(line)
            rho = line[0][0]
            theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(copy, pt1, pt2, (255, 0, 255), 3, cv2.LINE_AA)
    return copy, lines


def white_points_cloud_detection(image: np.ndarray, offset=0):
    # find a polynomial curve that fits the curves
    # the curve should be the same when steering at an angle, and should be obvious when the steering angle is 0
    # (the lines
    # intersect or are perfect parallel)
    # LOOK 2/3 THE IMAGE
    img = image[offset:]
    # point cloud calculation
    middle = img.shape[1] // 2
    point_cloud_left = []
    point_cloud_right = []
    for y in range(img.shape[0] - 1, -1, -1):
        row = img[y, :]
        left_part = row[:len(row) // 2]
        right_part = row[len(row) // 2:]
        white_points_left = np.nonzero(left_part == 255)[0]
        white_points_right = np.nonzero(right_part == 255)[0]

        if len(white_points_left) != 0:
            left_most_left_x = white_points_left[0]
            right_most_left_x = white_points_left[len(white_points_left) - 1]

            point_x_mean = (left_most_left_x + right_most_left_x) // 2
            point_cloud_left.append((y, img.shape[1] // 2 - point_x_mean))

        if len(white_points_right) != 0:
            left_most_right_x = white_points_right[0]
            right_most_right_x = white_points_right[len(white_points_right) - 1]

            point_x_mean = (left_most_right_x + right_most_right_x) // 2
            point_cloud_right.append((y, point_x_mean))
    #
    # for x in range(middle, img.shape[1]):
    #     # retrieve the column
    #     column = img[:, x]
    #     white_points_y = np.nonzero(column == 255)[0]
    #
    #     if len(white_points_y) == 0:
    #         continue
    #     if len(point_cloud_right) == 0:
    #         first_right_x = x
    #     # we only care about the first occurrence from the top
    #     point_y = white_points_y[0]
    #     point_cloud_right.append((x - first_right_x, img.shape[0] - point_y))
    # for x in range(middle - 1, -1, -1):
    #     # retrieve the column
    #     column = img[:, x]
    #     white_points_y = np.nonzero(column == 255)[0]
    #     if len(white_points_y) == 0:
    #         continue
    #     if len(point_cloud_left) == 0:
    #         first_left_x = x
    #     point_y = white_points_y[0]
    #     point_cloud_left.append((first_left_x - x, img.shape[0] - point_y))
    return point_cloud_left, point_cloud_right


def poly_fit(points, degree: int):
    x_axis = [point[0] for point in points]
    y_axis = [point[1] for point in points]
    coef = np.polyfit(x_axis, y_axis, degree)
    return coef


def plot_function(points, degree: int):
    # x_axis = [point[0] for point in points]
    # y_axis = [point[1] for point in points]
    poly_coef = poly_fit(points, degree)
    poly_function = np.poly1d(poly_coef)
    return poly_function


def radius_of_curvature(poly_function, point):
    first_derivative = poly_function.deriv(1)
    second_derivative = poly_function.deriv(2)
    result = pow(1 + pow(first_derivative(point), 2), 1.5) / abs(second_derivative(point))
    return result


def angle(height, radius):
    alpha = np.arctan(np.tan(height / radius)) * 180 / np.pi
    beta = (180 - alpha)//2
    return 90 - beta


def steering_angle(image: np.ndarray):
    triangle_heights = np.asarray([i * 30 for i in range(1, 10)])
    img = deepcopy(image)
    cloud_left, cloud_right = white_points_cloud_detection(img, 30)
    left_function, right_function = plot_function(cloud_left, 2), plot_function(cloud_right, 2)

    left_radius = radius_of_curvature(left_function, 50)
    right_radius = radius_of_curvature(right_function, 50)

    left_steering_angles = angle(triangle_heights, left_radius)
    right_steering_angles = angle(triangle_heights, right_radius)

    # NOT SO GOOD
    if left_radius > 1500 or right_radius > 1500:
        curve_direction = 0
    else:
        if left_radius < right_radius:
            curve_direction = -1
        else:
            curve_direction = 1
    final_angle = curve_direction * np.mean((np.mean(left_steering_angles), np.mean(right_steering_angles)))
    return final_angle


def offset_from_center(image: np.ndarray):
    """
    Compute the average offset from center in the lane
    :param image: np.ndarray
    :return: int
    """
    offsets = []
    img = deepcopy(image)[image.shape[0]-20:]
    cloud_left, cloud_right = white_points_cloud_detection(img)
    cloud_left.reverse()
    cloud_right.reverse()
    for i in range(img.shape[0]):
        try:
            offsets.append(cloud_right[i][1] - cloud_left[i][1])
        except Exception:
            continue
    if len(offsets) == 0:
        return None
    return np.mean(np.asarray(offsets))
