import os
import glob
import random
import time
import json
from time import perf_counter

import cv2
import torch
import numpy as np
import onnxruntime as ort

from lightglue import SuperPoint

import logging as logger


# # Paths to the onnx models
# ALIKED_LG_ONNX_PATH = "app/stitching_v2/lib/weights/onnx/aliked_lightglue.onnx"
# SUPERPOINT_ONNX_PATH = "/Users/haoyu/clobotics/LightGlue-ONNX/weights/superpoint.onnx"
# SUPERPOINT_ONNX_PATH = "/Users/haoyu/clobotics/LightGlue-ONNX/weights/superpoint.fp16.onnx"
SUPERPOINT_ONNX_PATH = "/Users/haoyu/clobotics/LightGlue-ONNX/weights/superpoint_batch_sim.onnx"
# SUPERPOINT_ONNX_PATH = "/Users/haoyu/Downloads/superpoint.onnx"




class FeatExtractor:
    def __init__(
        self,
        feat_type: str = "superpoint",
        max_keypoints: int = 2048,
        device: str = "cpu",
    ):
        self.device = device
        if feat_type == "superpoint":
            self.extractor = SuperPoint(
                max_num_keypoints=max_keypoints).eval().to(device)
        else:
            raise ValueError(f"Unknown feature type: {feat_type}")

    def run(self, data):
        return self.extractor.extract(data.to(self.device))


class Extractor:
    def __init__(
        self,
        model_path: str = SUPERPOINT_ONNX_PATH,
        providers=None
    ):
        assert os.path.exists(model_path), f"Path {model_path} does not exist."

        if providers is None:
            providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
            if os.name == "posix": # for macos
                providers = ["CoreMLExecutionProvider"]

        sess_options = ort.SessionOptions()
        # offline optimization not be considered, pass it
        self.model = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        # get input name and output shapes
        self.input_name = self.model.get_inputs()[0].name
        self.out_shapes = [out.shape for out in self.model.get_outputs()]

    def run(self, imgs: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        bs = imgs.shape[0]
        try:
            kpts, scores, descs = self.model.run(None, {self.input_name: imgs})
            # raise Exception("Test Error model inference")
        except Exception as e:
            logger.error(f"Error: {e}")
            return [np.zeros((bs, *shape[1:]), dtype=np.float32) for shape in self.out_shapes]

        return kpts, scores, descs


# given a list of images which type is np.ndarray, using batch onnx model to extract the keypoints
def extract_keypoints(images: list, nn_model: Extractor):
    # images shape is [[H, W, C], [H, W, C], ...]
    # convert to gray scale
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255. for img in images]
    # transpose to [B, 1, H, W]
    img = np.stack(images, axis=0)
    img = img[:, None].astype(np.float32)
    return nn_model.run(img)


def draw_points(image, points):
    for i in range(points.shape[0]):
        x, y = points[i]
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
    return image


if __name__ == "__main__":
    # Load the onnx model
    extractor = Extractor()

    # Load the image
    img0 = cv2.imread(
        "/Users/haoyu/clobotics/LightGlue-ONNX/assets/DSC_0410.JPG")
    img1 = cv2.imread(
        "/Users/haoyu/clobotics/LightGlue-ONNX/assets/DSC_0411.JPG")


    print(img0.shape, img1.shape)

    img_list = [img0, img1, img0, img1]

    kpts, scores, descs = extract_keypoints(img_list, extractor)
    for im, kpts in zip(img_list, kpts):
        im = draw_points(im, kpts)
        cv2.imshow("img", im)
        cv2.waitKey(0)

    print(kpts, "=======")
    print(kpts.shape, scores.shape, descs.shape)


    quit()

    # stack
    img = np.stack([img0[None], img1[None], img0[None], img1[None]], axis=0)
    # convert to float32
    img = img.astype(np.float32)

    # img = torch.tensor(img).unsqueeze(0).float()

    # # warm up 10 times
    # for _ in range(10):
    #     kpts, scores, descs = extractor.run(img)

    t0 = perf_counter()
    # Run the model
    kpts, scores, descs = extractor.run(img)
    t1 = perf_counter()
    print(f"Time: {(t1 - t0) * 1000} ms")

    for i in kpts:
        print(i)

    # diff 1 vs 3
    print(np.abs(kpts[0] - kpts[2]).mean())

    quit()

    print(kpts.shape, scores.shape, descs.shape)
    print(kpts, "=======")

    torch_model = FeatExtractor()
    output = torch_model.run(torch.tensor(img[0:1]).float())
    torch_kpt = output["keypoints"].numpy()
    torch_score = output["keypoint_scores"].numpy()
    torch_desc = output["descriptors"].numpy()

    # convert it into int64
    kpts = kpts.astype(np.int64)
    scores = scores.astype(np.float32)
    descs = descs.astype(np.float32)

    torch_kpt = torch_kpt.astype(np.int64)
    torch_score = torch_score.astype(np.float32)
    torch_desc = torch_desc.astype(np.float32)

    # test value difference
    print(np.abs(kpts - torch_kpt).mean())
    print(np.abs(scores - torch_score).mean())
    print(np.abs(descs - torch_desc).mean())

    # quit()

    # draw those keypoints on the image
    h, w = img0.shape
    for i in range(kpts.shape[1]):
        x, y = kpts[0][i]
        cv2.circle(img0_raw, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.imshow("img0", img0_raw)
    cv2.waitKey(0)

    # Save the output
    np.save("data/kpts.npy", kpts)
    np.save("data/scores.npy", scores)
    np.save("data/descs.npy", descs)

    print(kpts, scores, descs)
