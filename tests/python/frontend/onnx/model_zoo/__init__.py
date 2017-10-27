"""Store for onnx examples and common models."""
from __future__ import absolute_import as _abs
import os
import torchvision.models as models
from .super_resolution import get_super_resolution, save_onnx_super_resolution
from .squeezenet import save_onnx_squeezenet1_1

__all__ = ['super_resolution', 'squeezenet1_1', 'vgg11']

def _as_abs_path(fname):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(cur_dir, fname)

# a pair of onnx pb file and corresponding nnvm symbol
save_onnx_super_resolution()
super_resolution = (_as_abs_path('super_resolution.onnx'), get_super_resolution())
save_onnx_squeezenet1_1()
squeezenet1_1 = (_as_abs_path('squeezenet1_1.onnx'), None)

# pytorch model zoo
def save_onnx_model(model, name):
    from torch.autograd import Variable
    import torch
    batch_size = 1    # just a random number

    # Input to the model
    x = Variable(torch.randn(batch_size, 3, 224, 224), requires_grad=True)

    # Export the model
    model_path = os.path.join(os.path.dirname(__file__), name + '.onnx')
    torch_out = torch.onnx._export(torch_model,             # model being run
                                   x,                       # model input (or a tuple for multiple inputs)
                                   model_path,              # where to save the model (can be a file or file-like object)
                                   export_params=True)      # store the trained parameter weights inside the model file

save_onnx_model(models.vgg11(True), 'vgg11')
vgg11 = (_as_abs_path('vgg11'), None)
