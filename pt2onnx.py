from onnx import version_converter
import os

import torch
from networks import yolo_v11_n
import onnxruntime as ort
from argparse import ArgumentParser



def main():
    parser = ArgumentParser()
    parser.add_argument('--input_size', default=640, type=int)
    parser.add_argument('--nc', default=3, type=int)
    parser.add_argument('--model', required=True)

    args = parser.parse_args()
    assert args.model.endswith('.pt'), f"Model file {args.model} is not a .pt file"
    
    input_tensor = torch.rand((1, 3, args.input_size, args.input_size), dtype=torch.float32)

    model = yolo_v11_n(args.nc)
    model.load_state_dict(torch.load(args.model,weights_only=False))
    out_dir = os.path.dirname(args.model) 

    torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    os.path.join(out_dir, os.path.basename(args.model).replace('pt', 'onnx')),        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    do_constant_folding=True,
    opset_version= 12             # True or False to select the exporter to use
    
    )
if __name__ == "__main__":
    main()