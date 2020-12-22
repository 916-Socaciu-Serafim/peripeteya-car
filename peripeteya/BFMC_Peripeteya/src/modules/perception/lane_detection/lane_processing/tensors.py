class Tensor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._fire_history = []
        self._fired = False

    def toggle_fire(self, fire):
        self._fired = fire
        self.add_to_history(self._fired)

    def add_to_history(self, fire: bool):
        if len(self._fire_history) > 20:
            self._fire_history.pop(0)
        self._fire_history.append(fire)

    def __str__(self):
        return f"Tensor: width: {self.width}, height: {self.height}, fired: {self._fired}"


class TensorPair:
    def __init__(self, left_tensor: Tensor, right_tensor: Tensor, distance: int, position: tuple):
        self.left_tensor = left_tensor
        self.right_tensor = right_tensor
        self.init_distance = 0
        self.distance = distance
        self._positions_history = []
        self._distances_history = []
        self.position = position

    def update_position(self, new_position: tuple):
        self.position = new_position
        self._positions_history.append(self.position)

    def update_distance(self, new_distance: int):
        self.distance = new_distance
        self._distances_history.append(self.distance)

    def tensors(self):
        return [self.left_tensor, self.right_tensor]

    def __str__(self):
        prompt = f"Left tensor: {str(self.left_tensor)}\n"
        prompt += f"Distance: {self.distance}, Position: {self.position}\n"
        prompt += f"Right tensor: {str(self.right_tensor)}\n"
        return prompt
