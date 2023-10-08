import argparse
import time
from pathlib import Path

import numpy as np
import openvino as ov
from tqdm import tqdm
from vs_detection.src.utils import preprocess_image, prepare_input_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=str, default='ov',
                        choices=['ov', 'onnx'])
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--img_h', type=int, default=640)
    return parser.parse_args()


def measure_openvino(img_w, img_h, model_path, n=100):
    model_path = Path(model_path)

    img = np.random.randint(255, size=(img_h, img_w, 3)).astype(np.uint8)

    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, 'CPU')

    output_blob = compiled_model.output(0)
    preprocessed_img, orig_img = preprocess_image(img, (img_h, img_w))
    input_tensor = prepare_input_tensor(preprocessed_img)

    times = []
    for _ in tqdm(range(n)):
        start = time.perf_counter()
        _ = compiled_model(input_tensor)[output_blob]
        end = time.perf_counter()
        times.append(end - start)
    print('Avg time:', np.mean(times))
    print('FPS:', 1 / np.mean(times))


def main():
    args = parse_args()
    if args.framework == 'ov':
        measure_openvino(args.img_w, args.img_h, args.weights)
    elif args.framework == 'onnx':
        measure_openvino(args.img_w, args.img_h, args.weights)
    else:
        raise RuntimeError(f'Unknown framework: {args.framework}')


if __name__ == '__main__':
    main()
