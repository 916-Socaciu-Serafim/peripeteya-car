import cv2
import numpy as np

from src.modules.perception.lane_detection.test_images.image_processing import lane_start_points, \
    split_horizontally, generate_images, init_tensors_image, proper_size
from src.modules.perception.lane_detection.lane_processing.tensors import TensorPair, Tensor
from src.modules.perception.lane_detection.video_processing.video_processing import read_video_to_frames


def find_nearest(x, collection: list, tolerance=0):
    if len(collection) == 0:
        raise Exception
    distances = [abs(y - x) for y in collection]
    minimum_distance = min(distances)
    index = distances.index(minimum_distance)
    if minimum_distance > tolerance:
        raise Exception("Minimum distance to nearest element is greater the tolerance")
    return collection[index]
    pass


class TensorsMaster:
    def __init__(self, n: int, width: int, height: int, distance_tolerance=20):
        self._number_of_pairs = n
        self._tensors_pairs = []
        self._tensor_width = width
        self._tensor_height = height

    def initialize_tensors(self, zero_angle_image: np.ndarray):
        """
        Initialize the positions of the tensor in the image
        :param zero_angle_image: image that will be used as reference for the next image (this is where the tensors are
         placed first, if tensor master can not place all tensors then error occures)
        :return: nothing
        """
        # split the image according to the number of tensors pairs
        image_slices = split_horizontally(zero_angle_image, self._number_of_pairs)
        for i in range(len(image_slices)):
            img_slice = image_slices[i]
            accumulation_points = lane_start_points(img_slice)
            print(accumulation_points)
            cv2.imshow("slice", img_slice)
            cv2.waitKey(0)
            if len(accumulation_points) != 2:
                raise Exception("Image is not good for initialization, need exactly 2 accumulation points!")
            tensor_y_pos = img_slice.shape[0] // 2 + zero_angle_image.shape[0] // self._number_of_pairs * i
            tensor_left_x_pos = accumulation_points[0]
            tensor_right_x_pos = accumulation_points[1]
            distance = tensor_right_x_pos - tensor_left_x_pos
            tensor_x_pos = (tensor_left_x_pos + tensor_right_x_pos)//2
            tensor_left = Tensor(self._tensor_width, self._tensor_height)
            tensor_right = Tensor(self._tensor_width, self._tensor_height)
            tensor_left.toggle_fire(True)
            tensor_right.toggle_fire(True)
            tensors_pair = TensorPair(tensor_left, tensor_right, distance, (tensor_x_pos, tensor_y_pos))
            tensors_pair.init_distance = distance
            self._tensors_pairs.append(tensors_pair)
        pass

    def update_tensors(self, image: np.ndarray):
        # update the positions of the tensors pairs horizontally
        image_slices = split_horizontally(image, self._number_of_pairs)
        for i in range(len(image_slices)):
            img_slice = image_slices[i]
            cv2.imshow(f"slice{i}", img_slice)
            tensor_pair = self._tensors_pairs[i]
            left_tensor_x_pos = tensor_pair.position[0] - tensor_pair.distance//2
            right_tensor_x_pos = left_tensor_x_pos + tensor_pair.distance
            accumulation_points = lane_start_points(img_slice)
            print(accumulation_points)
            # todo: need a binary decision tree here
            try:
                next_left_tensor_pos_x = find_nearest(left_tensor_x_pos, accumulation_points, 15)
                tensor_pair.left_tensor.toggle_fire(True)
            except Exception:
                next_left_tensor_pos_x = left_tensor_x_pos
                tensor_pair.left_tensor.toggle_fire(False)
            try:
                next_right_tensor_pos_x = find_nearest(right_tensor_x_pos, accumulation_points, 15)
                tensor_pair.right_tensor.toggle_fire(True)
            except Exception:
                next_right_tensor_pos_x = right_tensor_x_pos
                tensor_pair.right_tensor.toggle_fire(False)
            new_distance = next_right_tensor_pos_x - next_left_tensor_pos_x
            new_position = ((next_left_tensor_pos_x + next_right_tensor_pos_x) // 2, tensor_pair.position[1])
            tensor_pair.update_distance(new_distance)
            tensor_pair.update_position(new_position)
            self._tensors_pairs[i] = tensor_pair
        pass

    def mean_delta_change(self):
        # compute the mean delta change of the positions of all the tensors.
        # return integer (positive or negative)
        pass

    def tensors_pairs(self):
        return self._tensors_pairs

    def __iter__(self):
        return iter(self._tensors_pairs)


master = TensorsMaster(3, 20, 20, 20)
img = init_tensors_image()
results = generate_images(img)
print(results["warped_thresh"].shape)
cv2.imshow("warped", results["warped_thresh"])
cv2.waitKey(0)
frames = read_video_to_frames("../image_processing/movie_004.mp4")[10:]
results = generate_images(proper_size(frames[0]))
master.initialize_tensors(results["warped_thresh"])
for img in frames[1:]:
    img = proper_size(img)
    results = generate_images(img)
    master.update_tensors(results["warped_thresh"])
    warped = np.dstack([results["warped_thresh"], results["warped_thresh"], results["warped_thresh"]])
    for tensor_pair in master.tensors_pairs():
        # draw the tensor on to the warped image
        pass
    cv2.imshow("warped", warped)
    print(*master, sep='\n')
    cv2.waitKey(0)

# find_nearest_test
# x = 10
# test_list = [29, 7, 13, 13, 7]
# near = find_nearest(x, test_list, 21)
# print(near)*