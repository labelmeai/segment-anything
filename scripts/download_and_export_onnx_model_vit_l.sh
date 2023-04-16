#!/bin/bash -e

checkpoint_path=../models/sam_vit_l_0b3195.pth
if [[ "$(md5sum $checkpoint_path | awk '{print $1}')" =~ ^0b3195 ]]; then
  echo "==> Model checkpoint is already downloaded"
else
  echo "==> Downloading model checkpoint"
  set -x
  curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth \
    -o $checkpoint_path
  { set +x; } 2>/dev/null
fi

echo "==> Exporting onnx model from checkpoint"
set -x
python export_onnx_model.py \
  --checkpoint $checkpoint_path \
  --model-type vit_l \
  --encoder-output ../models/sam_vit_l_0b3195.encoder.onnx \
  --decoder-output ../models/sam_vit_l_0b3195.decoder.onnx \
  --quantize-encoder-out ../models/sam_vit_l_0b3195.quantized.encoder.onnx \
  --quantize-decoder-out ../models/sam_vit_l_0b3195.quantized.decoder.onnx \
  --return-single-mask
{ set +x; } 2>/dev/null
