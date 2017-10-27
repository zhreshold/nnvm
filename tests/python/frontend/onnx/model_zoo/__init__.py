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
vgg11 = (models.vgg11(True), None)
