import torch
import onnx
import numpy as np
import onnxruntime
import time

from SuperPointPretrainedNetwork.demo_superpoint import SuperPointNet

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('height', type=int, help='Input height.')
parser.add_argument('width', type=int, help='Input width.')

args = parser.parse_args()

print(f"Input shape is [1,1,{args.height},{args.width}] \n")

#export .pt to .onnx
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--------Inferring on {device}--------")

net = SuperPointNet()
net.load_state_dict(torch.load('SuperPointPretrainedNetwork/superpoint_v1.pth',
                        map_location=lambda storage, loc: storage))
net.eval()

net.to(device)
start = time.time()
dummy_input = torch.randn(1,1,args.height,args.width).to(device)
torch_out = net(dummy_input)
end = time.time()
print(f"Inference of Pytorch model used {end - start} seconds\n")

torch.onnx.export(net, dummy_input, "../weights/superpoint.onnx", input_names=[ "input" ], output_names=[ "semi", "coarse_desc" ])

#check .onnx
model = onnx.load("../weights/superpoint.onnx")
onnx.checker.check_model(model)

#compare ouput from .onnx ans .pt
ort_session = onnxruntime.InferenceSession("../weights/superpoint.onnx", 
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
onnx_device = onnxruntime.get_device()
print(f"Current ONNX device: {onnx_device}")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
start = time.time()
ort_outs = ort_session.run(None, ort_inputs)
end = time.time()
print(f"Inference of ONNX model used {end - start} seconds\n")

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")