#!/bin/bash -e

checkpoint_path=../models/sam_vit_h_4b8939.pth
if [[ "$(md5sum $checkpoint_path | awk '{print $1}')" =~ ^4b8939 ]]; then
  echo "==> Model checkpoint is already downloaded"
else
  echo "==> Downloading model checkpoint"
  set -x
  curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -o $checkpoint_path
  { set +x; } 2>/dev/null
fi

echo "==> Exporting onnx model from checkpoint"
set -x
python export_onnx_model.py \
  --checkpoint $checkpoint_path \
  --model-type vit_h \
  --encoder-output ../models/sam_vit_h_4b8939.encoder.onnx \
  --decoder-output ../models/sam_vit_h_4b8939.decoder.onnx \
  --quantize-encoder-out ../models/sam_vit_h_4b8939.quantized.encoder.onnx \
  --quantize-decoder-out ../models/sam_vit_h_4b8939.quantized.decoder.onnx
{ set +x; } 2>/dev/null
