"""
From Alexander Kharin https://github.com/Alexankharin
https://github.com/Alexankharin/3dfilms
"""
import os
import time
import traceback

import cv2
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from ...image_utils import save_image


class Converter:
    def __init__(self):
        self.model = None

    def load(self, config):
        bus2D = tf.keras.layers.UpSampling2D(
            size=(2, 2), data_format=None, interpolation="bilinear"
        )
        custom_objects = {
            "BilinearUpSampling2D": bus2D,
            "depth_loss_function": depth_loss_function,
            "ssim_loss": ssim_loss,
        }
        model_path = "./models/nueral2d3d.h5"
        assert os.path.isfile(
            model_path
        ), f"No model found at {os.path.abspath(model_path)}"
        self.model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects
        )
        self.model.compile(loss="mse", optimizer="Adam")

    def convert_image(self, imagepath, output_path):
        converter = self.model
        img = cv2.imread(imagepath)
        # flip channels from bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.stack([img])
        shapeout = (
            int(img.shape[1] // 32 - 0.000001) * 32 + 32,
            (int(img.shape[2] // 32 - 0.000001) * 32 + 32),
            3,
        )
        img = img[:, : shapeout[0], : shapeout[1], : shapeout[2]]
        CC = np.expand_dims(
            np.transpose(
                np.indices((img.shape[1], img.shape[2])) / img.shape[1], (1, 2, 0)
            ),
            0,
        )
        predictedimage = converter.predict([img, CC]).astype(np.uint8)
        predictedimage: np.ndarray = predictedimage[0, :, :, :].clip(0, 255)
        img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)
        predictedimage = cv2.cvtColor(predictedimage, cv2.COLOR_RGB2BGR)
        output_image = np.concatenate([img, predictedimage], axis=1)
        save_image(output_image, output_path)

    def convert_video(self, videoname, outname, size=None):
        assert os.path.isfile(
            videoname
        ), f"No video found at {os.path.abspath(videoname)}"
        modelconverter = self.model
        vidcap = cv2.VideoCapture(videoname)
        success, image = vidcap.read()
        print(f"Video read success is {success}")
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        if size is None:
            shapeout = (
                int(np.ceil(image.shape[1] / 32) * 32) * 2,
                int(np.ceil(image.shape[0] / 32) * 32),
            )
        else:
            shapeout = (size[0] // 32 * 32 * 2, size[1] // 32 * 32)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Loaded video at {videoname} with fps {fps}")
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        out = cv2.VideoWriter(
            outname,
            fourcc,
            int(fps),
            shapeout,
        )
        print("Writing video to ", os.path.abspath(outname))
        startflag = 0
        count = 0
        ctr = 0
        timein = time.time()
        with tqdm(total=total_frames) as pbar:
            while success and image is not None:
                try:
                    print(count)
                    count += 1
                    ctr = ctr + 1
                    imageU = tf.image.resize_with_pad(
                        image, shapeout[1], shapeout[0] // 2
                    )
                    U = imageU
                    CC = np.expand_dims(
                        np.transpose(
                            np.indices((U.shape[0], U.shape[1])) / U.shape[0], (1, 2, 0)
                        ),
                        0,
                    )
                    D = modelconverter.predict([np.stack([U / 255]), CC])[
                        0, :, :, :
                    ].clip(0, 1)
                    frame = np.concatenate([U, D * 255], 1).astype(np.uint8)
                    if startflag == 0:
                        startflag = 1
                    out.write(frame)
                    if ctr % (fps) == 0:
                        print(
                            ctr // (fps),
                            "  s  ",
                            time.time() - timein,
                            " sec per videosecond",
                        )
                        timein = time.time()
                        # break
                    # if ctr==10:
                    #  break
                    success, image = vidcap.read()
                except:
                    print(print(traceback.format_exc()))
                    break
                pbar.update(1)
        vidcap.release()
        out.release()


def depth_loss_function(y_true, y_pred, theta=0.1, max_depth_val=1000.0 / 10.0):
    # vggloss
    lvgg = K.mean(K.abs(lossmodel(ytrue) - lossmodel(ypred)))
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta
    w4 = 0.1

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth)) + w4 * lvgg


# keras.backend.normalize_data_format=normalize_data_format
def ssim_loss(y_true, y_pred):
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1)) * 0.5, 0, 1)
    # vggloss
    lvgg = K.clip(K.mean(K.abs(lossmodel(y_true) - lossmodel(y_pred))), 0, 1)
    return l_ssim + tf.keras.losses.MAE(y_true, y_pred) + lvgg
