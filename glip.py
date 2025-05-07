import os
import sys
import numpy as np
import torch
import requests
from time import time
from io import BytesIO
from pathlib import Path
from PIL import Image

# Optional plotting
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# Make sure GLIP is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

# Constants
GLIP_DIR = os.path.join(Path(__file__).parent.resolve(), "GLIP")
WEIGHT_PATH = os.path.join(GLIP_DIR, "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth")
CONFIG_FILE = os.path.join(GLIP_DIR, "configs/pretrain/glip_Swin_T_O365_GoldG.yaml")
REMOTE_PATH = "https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth"


class GLIPPrediction:
    def __init__(self, bboxes, scores, labels) -> None:
        self.bboxes = bboxes
        self.scores = scores
        self.labels = labels
        self.n = len(bboxes)


class GLIP:
    def __init__(self) -> None:
        self._download_weight()
        self._init_cfg(cfg)
        self._check_gpu()
        self._init_glip_model()

    def _init_cfg(self, cfg):
        self.cfg = cfg
        self.cfg.local_rank = 0
        self.cfg.num_gpus = 1
        self.cfg.merge_from_file(CONFIG_FILE)
        self.cfg.merge_from_list(["MODEL.WEIGHT", WEIGHT_PATH])
        self.cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    def _download_weight(self):
        if not os.path.exists(WEIGHT_PATH):
            Path(WEIGHT_PATH).parent.mkdir(parents=True, exist_ok=True)
            os.system(f"wget {REMOTE_PATH} -O {WEIGHT_PATH}")

    def _check_gpu(self):
        try:
            print("Check GPU info:")
            print("CUDA available: {}".format(torch.cuda.is_available()))
            print("CUDA version: {}".format(torch.version.cuda))
            print("GPU count: {}".format(torch.cuda.device_count()))
            print("GPU name: {}".format(torch.cuda.get_device_name(0)))
        except Exception as e:
            raise ValueError("Fail to check GPU info due to {}".format(e))

    def _init_glip_model(self):
        print("Initializing GLIP Model...")
        start = time()
        self.glip_model = GLIPDemo(
            self.cfg,
            min_image_size=800,
            confidence_threshold=0.7,
            show_mask_heatmaps=False,
        )
        print("GLIP Model Ready in {:.2f} seconds".format(time() - start))

    def predict(self, image, query, debug=False) -> GLIPPrediction:
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        predictions = self.glip_model.inference(image, query)
        bboxes = predictions.bbox.tolist()
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        if debug:
            print("bboxes: {}".format(bboxes))
            print("scores: {}".format(scores))
            print("labels: {}".format(labels))
        class_names = query.split(" ")
        classes = [class_names[label - 1] for label in labels]
        return GLIPPrediction(bboxes, scores, classes)

    def run_and_save(self, image, prompt, output_path="output.jpg"):
        result_bgr, _ = self.glip_model.run_on_web_image(image, prompt, 0.5)
        result_rgb = result_bgr[:, :, [2, 1, 0]]
        result_image = Image.fromarray(result_rgb)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_image.save(output_path)
        print(f"[✓] Saved result to: {output_path}")


def load_image(source):
    if source.startswith("http"):
        response = requests.get(source)
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        pil_image = Image.open(source).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


if __name__ == "__main__":
    # Set your image source and prompt
    image_path = "/home/vmukti/Pictures/car.jpg"  # or a URL
    prompt = "white car in the leftmost side"
    output_path = "outputs/output.jpeg"

    # Run GLIP
    glip = GLIP()
    image = load_image(image_path)
    glip.run_and_save(image, prompt, output_path)
