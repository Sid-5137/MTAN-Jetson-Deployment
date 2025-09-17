#!/usr/bin/env python3
"""
MTAN Inference Script
Optimized for Jetson deployment
Supports PyTorch, ONNX, and TensorRT models
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from config import Config, CITYSCAPES_CLASSES, CITYSCAPES_COLORS
from models import create_mtan_model
from data import CityscapesVisualization

# -------------------------------
# MTAN Inference Engine
# -------------------------------
class MTANInference:
    def __init__(self, model_path: str, config: Config, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model_type = self._detect_model_type()
        self.model = self._load_model()

        # Normalization
        self.mean = np.array(config.data.normalize_mean).reshape(1, 3, 1, 1)
        self.std = np.array(config.data.normalize_std).reshape(1, 3, 1, 1)

        if self.model_type == 'pytorch':
            self.mean = torch.tensor(self.mean).to(self.device)
            self.std = torch.tensor(self.std).to(self.device)

    def _detect_model_type(self):
        ext = Path(self.model_path).suffix.lower()
        if ext == '.pth':
            return 'pytorch'
        elif ext == '.onnx':
            return 'onnx'
        elif ext == '.trt':
            return 'tensorrt'
        else:
            raise ValueError(f"Unsupported model format: {ext}")

    def _load_model(self):
        if self.model_type == 'pytorch':
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model = create_mtan_model(self.config)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model.to(self.device).eval()
            return model

        elif self.model_type == 'onnx':
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = session.get_inputs()[0].name
            return session

        elif self.model_type == 'tensorrt':
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            with open(self.model_path, 'rb') as f:
                engine_data = f.read()
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            self._allocate_trt_memory(engine)
            return {'engine': engine, 'context': context}

    def _allocate_trt_memory(self, engine):
        import pycuda.driver as cuda
        self.trt_inputs, self.trt_outputs, self.trt_bindings = [], [], []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.trt_bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.trt_inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.trt_outputs.append({'host': host_mem, 'device': device_mem})

    def preprocess(self, image: np.ndarray) -> Any:
        h, w = self.config.model.input_size
        image = cv2.resize(image, (w, h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[None, ...]  # NCHW
        if self.model_type == 'pytorch':
            tensor = torch.from_numpy(image).to(self.device)
            tensor = (tensor - self.mean) / self.std
            return tensor
        return (image - self.mean) / self.std

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        x = self.preprocess(image)
        if self.model_type == 'pytorch':
            with torch.no_grad():
                out = self.model(x)
                seg = torch.softmax(out['segmentation'], 1).argmax(1).cpu().numpy()[0]
                depth = out['depth'].cpu().numpy()[0, 0]
        elif self.model_type == 'onnx':
            seg_logits, depth_pred = self.model.run(None, {self.input_name: x.astype(np.float32)})
            seg = np.argmax(seg_logits, axis=1)[0]
            depth = depth_pred[0, 0]
        else:  # TensorRT
            import pycuda.driver as cuda
            np.copyto(self.trt_inputs[0]['host'], x.ravel())
            cuda.memcpy_htod(self.trt_inputs[0]['device'], self.trt_inputs[0]['host'])
            self.model['context'].execute_v2(bindings=self.trt_bindings)
            outputs = []
            for out_mem in self.trt_outputs:
                cuda.memcpy_dtoh(out_mem['host'], out_mem['device'])
                outputs.append(out_mem['host'].copy())
            b, h, w = 1, *self.config.model.input_size
            seg = np.argmax(outputs[0].reshape(b, self.config.model.num_classes_seg, h, w), axis=1)[0]
            depth = outputs[1].reshape(b, 1, h, w)[0, 0]
        return {'segmentation': seg, 'depth': depth}

    def postprocess(self, outputs: Dict[str, np.ndarray], size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        seg = cv2.resize(outputs['segmentation'], size, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(outputs['depth'], size, interpolation=cv2.INTER_LINEAR)
        return {'segmentation': seg, 'depth': depth}

    def visualize(self, image: np.ndarray, outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        seg_col = CityscapesVisualization.decode_segmap(outputs['segmentation'])
        depth_col = CityscapesVisualization.visualize_depth(outputs['depth'])
        overlay = cv2.addWeighted(image, 0.6, seg_col, 0.4, 0)
        return {'input': image, 'segmentation': seg_col, 'depth': depth_col, 'overlay': overlay}

# -------------------------------
# Video/Image Processing
# -------------------------------
def process_video(inference: MTANInference, input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        outputs = inference.predict(frame)
        outputs = inference.postprocess(outputs, (w, h))
        vis = inference.visualize(frame, outputs)
        out.write(vis['overlay'])
        pbar.update(1)
    cap.release()
    out.release()
    pbar.close()

def process_image(inference: MTANInference, input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(input_path)
    outputs = inference.predict(img)
    outputs = inference.postprocess(outputs, (img.shape[1], img.shape[0]))
    vis = inference.visualize(img, outputs)
    base = Path(input_path).stem
    cv2.imwrite(os.path.join(output_dir, f'{base}_seg.png'), vis['segmentation'])
    cv2.imwrite(os.path.join(output_dir, f'{base}_depth.png'), vis['depth'])
    cv2.imwrite(os.path.join(output_dir, f'{base}_overlay.png'), vis['overlay'])

# -------------------------------
# CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='./output')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = Config()
    engine = MTANInference(args.model, config, args.device)
    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        process_video(engine, str(input_path), os.path.join(args.output, 'output.mp4'))
    elif input_path.is_file():
        process_image(engine, str(input_path), args.output)
    elif input_path.is_dir():
        for ext in ['.jpg', '.jpeg', '.png']:
            for f in input_path.glob(f'*{ext}'):
                process_image(engine, str(f), args.output)

if __name__ == '__main__':
    main()
