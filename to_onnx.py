import torch
from utils.model import model_builder

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='saved_model/AHDD_resnet18.pt')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model_builder()

    model.load_state_dict(torch.load(args.path))
    
    num_layers = len(list(model.parameters()))  

    dummy_input = torch.rand(1, 1, 28, 28).to(device)
    print(f'input shape: {model(dummy_input).shape}')
    model = model.to(device)
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(num_layers) ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, "saved_model/AHDD_resnet.onnx", verbose=True, input_names=input_names, output_names=output_names)