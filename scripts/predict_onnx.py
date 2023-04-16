#!/usr/bin/env python3
import imgviz
import numpy as np

import onnxruntime


image = imgviz.io.imread("../notebooks/images/truck.jpg")

# encoder_path = "../models/sam_vit_h_4b8939.encoder.onnx"
# decoder_path = "../models/sam_vit_h_4b8939.decoder.onnx"
encoder_path = "../models/sam_vit_h_4b8939.quantized.encoder.onnx"
decoder_path = "../models/sam_vit_h_4b8939.quantized.decoder.onnx"

encoder_session = onnxruntime.InferenceSession(encoder_path)
decoder_session = onnxruntime.InferenceSession(decoder_path)

from segment_anything.utils.transforms import ResizeLongestSide

image_size = 1024

if 0:
    import torch
    from torch.nn import functional as F

    transform = ResizeLongestSide(image_size)
    input_image = transform.apply_image(image)

    input_image_torch = torch.as_tensor(input_image, device="cpu")
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (input_image_torch - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = image_size - h
    padw = image_size - w
    x = F.pad(x, (0, padw, 0, padh))
    x = x.numpy()
else:
    assert image.shape[1] > image.shape[0]
    scale = image_size / image.shape[1]
    x = imgviz.resize(
        image,
        height=int(round(image.shape[0] * scale)),
        width=image_size,
        backend="pillow",
    ).astype(np.float32)
    x = (x - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array(
        [58.395, 57.12, 57.375], dtype=np.float32
    )
    x = np.pad(x, ((0, image_size - x.shape[0]), (0, image_size - x.shape[1]), (0, 0)))
    x = x.transpose(2, 0, 1)[None, :, :, :]

output = encoder_session.run(output_names=None, input_feed={"x": x})
image_embedding = output[0]

input_point = np.array([[500, 375]])
input_label = np.array([1])

onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(
    np.float32
)

assert image.shape[1] > image.shape[0]
scale = image_size / image.shape[1]
new_height = int(round(image.shape[0] * scale))
new_width = image_size
onnx_coord = (
    onnx_coord.astype(float) * (new_width / image.shape[1], new_height / image.shape[0])
).astype(np.float32)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

decoder_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32),
}

masks, _, logits = decoder_session.run(None, decoder_inputs)
masks = masks > 0.0

for i in range(masks.shape[1]):
    imgviz.io.imsave(f"{i}.jpg", imgviz.label2rgb(masks[0, i], imgviz.rgb2gray(image)))
