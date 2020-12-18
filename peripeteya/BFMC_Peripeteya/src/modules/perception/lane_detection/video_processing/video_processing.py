import cv2


def read_video_to_frames(full_path):
    cap = cv2.VideoCapture(full_path)
    frames = []
    if not cap.isOpened():
        return []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cv2.destroyAllWindows()
    return frames


def write_frames_to_video(frames, file_name):
    """
    NOT WORKING
    :param frames:
    :param file_name:
    :return:
    """
    frame_size = frames[10].shape
    frame_width = int(frame_size[0])
    frame_height = int(frame_size[1])

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('hello.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 50, (frame_width, frame_height))

    for frame in frames:
        # Write the frame into the file 'output.avi'
        out.write(frame)
        cv2.waitKey(10)
        # When everything done, release the video capture and video write objects
    out.release()
#
#

