from functools import partial

import cv2
import numpy as np

k1 = 0.3358 / 2  # negative to remove barrel distortion
k2 = 0.5534 / 2
p1 = 0  # -.001#-.005;
p2 = 0  # -.001#.01;
focal_length_x = 500.0
focal_length_y = 500.0


def distort_vr(image: np.ndarray, **kwargs)->np.ndarray:
    left, right = split_image_h(image)
    left_dst = barrel_distort(left, **kwargs)
    right_dst = barrel_distort(right, **kwargs)
    dst = cv2.hconcat([left_dst, right_dst])
    return dst


def barrel_distort(
    image: np.ndarray,
    k1=0.0,
    k2=0.0,
    p1=0,
    p2=0.0,
    focal_length_x=0.0,
    focal_length_y=0.0,
) -> np.ndarray:
    width = image.shape[1]
    height = image.shape[0]
    distCoeff = np.zeros((4, 1), np.float64)
    # assume unit matrix for camera
    cam = np.eye(3, dtype=np.float32)

    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2.0  # define center y
    cam[0, 0] = focal_length_x
    cam[1, 1] = focal_length_y
    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2

    # here the undistortion will be computed
    dst = cv2.undistort(image, cam, distCoeff)
    return dst


def split_image_h(image):
    center = image.shape[1] // 2
    left = image[:, :center]
    right = image[:, center:]
    return left, right


def cat_images_h(left, right):
    return cv2.hconcat([left, right])


cardboard_distort = partial(
    barrel_distort,
    k1=k1,
    k2=k2,
    p1=p1,
    p2=p2,
    focal_length_x=focal_length_x,
    focal_length_y=focal_length_y,
)
