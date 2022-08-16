import torch
import os
import sys
import time
import numpy as np
from src.pynn.net.seq2seq import Seq2Seq
import argparse


class Encoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, seq, masks):
        encoder_out, mask = self.encoder(seq, masks)[0:2]
        return encoder_out, mask


class Decoder(torch.nn.Module):
    def __init__(self, decoder):
        self.decoder = decoder

    def forward(self):
        pass


parser = argparse.ArgumentParser(description='pynn')

parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-head', type=int, default=8)
parser.add_argument('--n-enc', type=int, default=4)
parser.add_argument('--n-dec', type=int, default=2)
parser.add_argument('--d-input', type=int, default=40)
parser.add_argument('--d-model', type=int, default=320)

parser.add_argument(
    '--unidirect', help='uni directional encoder', action='store_true')
parser.add_argument(
    '--incl-win', help='incremental window size', type=int, default=0)
parser.add_argument(
    '--time-ds', help='downsample in time axis', type=int, default=1)
parser.add_argument('--use-cnn', help='use CNN filters', action='store_true')
parser.add_argument('--freq-kn', help='frequency kernel', type=int, default=3)
parser.add_argument('--freq-std', help='frequency stride', type=int, default=2)
parser.add_argument(
    '--no-lm', help='not combine with LM in decoder', action='store_true')
parser.add_argument(
    '--shared-emb', help='sharing decoder embedding', action='store_true')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--emb-drop', type=float, default=0.0)
parser.add_argument('--label-smooth', type=float, default=0.1)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--teacher-force', type=float, default=1.0)

parser.add_argument('--max-len', help='max sequence length',
                    type=int, default=5000)
parser.add_argument(
    '--max-utt', help='max utt per partition', type=int, default=4096)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument(
    '--zero-pad', help='padding zeros to sequence end', type=int, default=0)
parser.add_argument('--spec-drop', help='argument inputs', action='store_true')
parser.add_argument(
    '--spec-bar', help='number of bars of spec-drop', type=int, default=2)
parser.add_argument(
    '--time-stretch', help='argument inputs', action='store_true')
parser.add_argument('--time-win', help='time stretch window',
                    type=int, default=10000)
parser.add_argument('--model-path', help='model saving path', default='model')

# options for exporting
parser.add_argument('--batch-size', type=int, default=32,
                    help="batch size for dummy input and test")
parser.add_argument(
    '--export-onnx', help='whether export the onnx model', action='store_true')
parser.add_argument('--encoder-onnx-dir',
                    help='the exported onnx encoder directory', default='model/')
parser.add_argument(
    '--test-onnx', help='whether to test the onnx model', action='store_true')
parser.add_argument('--iterations', type=int, default=100,
                    help='whether to test the onnx model')
parser.add_argument(
    '--test-pytorch', help='to test pytorch baseline', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    model = Seq2Seq(
        input_size=args.d_input,
        hidden_size=args.d_model,
        output_size=args.n_classes,
        n_enc=args.n_enc,
        n_dec=args.n_dec,
        n_head=args.n_head,
        unidirect=args.unidirect,
        incl_win=args.incl_win,
        time_ds=args.time_ds,
        use_cnn=args.use_cnn,
        freq_kn=args.freq_kn,
        freq_std=args.freq_std,
        lm=not args.no_lm,
        shared_emb=args.shared_emb,
        dropout=args.dropout,
        emb_drop=args.emb_drop)

    model.eval()
    print(model)

    # export encoder
    feature_size = args.d_input
    bz = args.batch_size
    seq_len = 1000

    if args.export_onnx:
        seqs = torch.randn(bz, seq_len, feature_size, dtype=torch.float32)
        masks = torch.ones((bz, seq_len), dtype=torch.int32)
        encoder = Encoder(model.encoder)

        output_onnx_dir = args.encoder_onnx_dir
        if not os.path.exists(output_onnx_dir):
            os.makedirs(output_onnx_dir)
        encoder_onnx_fp32_path = os.path.join(output_onnx_dir, "encoder.onnx")
        torch.onnx.export(encoder,
                          (seqs, masks),
                          encoder_onnx_fp32_path,
                          export_params=True,
                          opset_version=14,
                          do_constant_folding=True,
                          input_names=["seqs", "masks"],
                          output_names=["encoder_out", "masks_out"],
                          dynamic_axes={"seqs": {0: 'B', 1: 'T'},
                                        "masks": {0: 'B', 1: 'T'},
                                        "encoder_out": {0: 'B', 1: 'T'},
                                        "masks_out": {0: 'B', 1: 'T'}},
                          verbose=False)

        # convert_fp16
        try:
            import onnxmltools
            from onnxmltools.utils.float16_converter import convert_float_to_float16
        except ImportError:
            print('Please install onnxmltools!')
            sys.exit(1)
        encoder_onnx_model = onnxmltools.utils.load_model(
            encoder_onnx_fp32_path)
        encoder_onnx_model = convert_float_to_float16(encoder_onnx_model)
        encoder_onnx_fp16_path = os.path.join(
            output_onnx_dir, 'encoder_fp16.onnx')
        onnxmltools.utils.save_model(
            encoder_onnx_model, encoder_onnx_fp16_path)

        if args.test_onnx:
            import onnxruntime as rt
            EP_list = ['CUDAExecutionProvider']
            # test fp32
            encoder_ort_session = rt.InferenceSession(
                encoder_onnx_fp32_path, providers=EP_list)
            ort_inputs = {encoder_ort_session.get_inputs()[0].name: seqs.numpy(),
                          encoder_ort_session.get_inputs()[1].name: masks.numpy()}
            iterations = 100
            start = time.time()
            for i in range(iterations):
                ort_outs = encoder_ort_session.run(None, ort_inputs)
            end = time.time()
            print("ONNX FP32 average: {} ms".format(
                round((end-start)/iterations * 1000, 1)))

            # test fp16
            encoder_ort_session = rt.InferenceSession(
                encoder_onnx_fp16_path, providers=EP_list)
            ort_inputs = {encoder_ort_session.get_inputs()[0].name: seqs.numpy().astype(np.float16),
                          encoder_ort_session.get_inputs()[1].name: masks.numpy()}
            start = time.time()
            for i in range(iterations):
                ort_outs = encoder_ort_session.run(None, ort_inputs)
            end = time.time()
            print("ONNX FP16 average: {} ms".format(
                round((end-start)/iterations * 1000, 1)))

    if args.test_pytorch:
        half = True
        iterations = args.iterations
        model.to(device)
        model.eval()

        def test_model(half):
            seq = torch.randn(bz, seq_len, feature_size, dtype=torch.float32)
            masks = torch.ones((bz, seq_len), dtype=torch.int32)
            FP = "FP32"
            if half:
                model.half()
                seq = seq.to(torch.float16)
                FP = "FP16"
            encoder = Encoder(model.encoder)
            encoder.to(device)
            start = time.time()
            for _ in range(iterations):
                seq = seq.cuda()
                encoder(seq, masks)
            end = time.time()
            print("Pytorch {0} average: {1} ms" .format(
                FP, round(((end - start) / iterations * 1000), 1)))
        # test fp32
        test_model(half=False)
        test_model(half=True)
