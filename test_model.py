import torch
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np
from export_deca_trace import OUTPUT_NAMES
from dataset_deca import TestData, FAN


# Define image preprocessing
def preprocess_image(image_path, face_detector=FAN()):
    ds = TestData([image_path], face_detector=face_detector)
    image = ds[0].unsqueeze(0)
    return image


@torch.no_grad()
def _print(name, x):
    if isinstance(x, torch.Tensor):
        x = x[0].detach().cpu().numpy()
    while len(x.shape) > 1:
        x = x[0]
    x_str = [str(i)[:6] for i in x[:5]]
    print(f"{name}: {x.shape} {x_str}")


@torch.no_grad()
def test_jit_model(image_path, model_path, use_detail):
    image = preprocess_image(image_path).to(device)
    # Load the JIT model
    jit_model = torch.jit.load(model_path)
    jit_model.eval()

    # Run the model
    outputs = jit_model(image)
    output_names = OUTPUT_NAMES + (["detail"] if use_detail else [])
    print(f"JIT Model Outputs ({'with' if use_detail else 'without'} detail):")
    for name, output in zip(output_names, outputs):
        _print(name, output)


@torch.no_grad()
def test_onnx_model(image_path, model_path, use_detail):
    image = preprocess_image(image_path).numpy()

    # Load the ONNX model
    ort_session = ort.InferenceSession(model_path)

    # Run the model
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    output_names = OUTPUT_NAMES + (["detail"] if use_detail else [])
    print(f"ONNX Model Outputs ({'with' if use_detail else 'without'} detail):")
    for name, output in zip(output_names, ort_outs):
        _print(name, output)


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the saved models
jit_model_no_detail_path = "deca_no_detail.pt"
jit_model_detail_path = "deca_detail.pt"
onnx_model_no_detail_path = "deca_no_detail.onnx"
onnx_model_detail_path = "deca_detail.onnx"

# Path to the image you want to test
image_path = "../test-face-images/left-wink.png"

# Test the models
test_jit_model(image_path, jit_model_no_detail_path, use_detail=False)
test_onnx_model(image_path, onnx_model_no_detail_path, use_detail=False)
test_onnx_model(image_path, onnx_model_detail_path, use_detail=True)
test_jit_model(image_path, jit_model_detail_path, use_detail=True)
