# LAS_zoom



## Getting started
Pull forked `pynn`
```
git clone https://github.com/arunraman/pynn.git
git checkout gpu-benchmark
```

## Environment
```
cd pynn
docker run --gpus '"device=0"' -it -v $PWD:/ws nvcr.io/nvidia/pytorch:22.07-py3
pip install onnxmltools onnxruntime-gpu  # for onnx fp16 model and testing onnx gpu
cd /ws/benchmark
mkdir model/
```
Please refer to ```export.sh``` and ```export_seq2seq.py``` for reference.

### Export to Tensorrt
To export to Tensorrt, there's one line in the source code needs to be changed.
Please add ```mask = None``` [here](https://github.com/thaisonngn/pynn/blob/master/src/pynn/net/seq2seq.py#L34).  The ```pack_padded_sequence``` seems to keep the batch size, which does not allow to export a model with variable batch size. (You can export tensorrt engine if the batch size is fixed with this mask.)

```
polygraphy surgeon sanitize encoder.onnx --fold-constants -o encoder_folded.onnx
trtexec --onnx=encoder_folder.onnx
 --minShapes=seqs:32x1000x40,masks:32x1000
 --optShapes=seqs:32x1000x40,masks:32x1000
 --maxShapes=seqs:32x1000x40,masks:32x1000
--saveEngine=encoder_fp32.trt

trtexec --onnx=encoder_folded.onnx
 --minShapes=seqs:32x1000x40,masks:32x1000
 --optShapes=seqs:32x1000x40,masks:32x1000
 --maxShapes=seqs:32x1000x40,masks:32x1000
 --saveEngine=encoder_fp16.trt --fp16
```
