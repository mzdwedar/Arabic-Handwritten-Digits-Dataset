import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import numpy as np

import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="saved_model/AHDD_resnet.onnx")
    parser.add_argument('--image_path', type=str)
    args = parser.parse_args()


    onnx_model = onnx.load(args.model_path)

    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession("AHDD_resnet.onnx")

    img = Image.open(args.image_path).resize((224, 224))
    image = np.expand_dims(np.array(img, dtype=np.float32), axis=(0,1))

    outputs = ort_session.run(
        None,
        {"actual_input_1": image},
    )

    print(outputs[0].argmax())

