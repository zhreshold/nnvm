"""NNVM symbol corresponding to super_resolution.onnx example."""
from nnvm import sym

def get_super_resolution():
    factor = 3
    size = 224
    data = sym.Variable(name='9')
    conv1 = sym.conv2d(data, channels=64, kernel_size=(5, 5), padding=(2, 2), use_bias=False)
    relu1 = sym.relu(conv1 + sym.reshape(sym.Variable(name='2', shape=(64)), shape=(1, -1, 1, 1)))
    conv2 = sym.conv2d(relu1, channels=64, kernel_size=(3, 3), padding=(1, 1), use_bias=False)
    relu2 = sym.relu(conv2 + sym.reshape(sym.Variable(name='4', shape=(64)), shape=(1, -1, 1, 1)))
    conv3 = sym.conv2d(relu2, channels=32, kernel_size=(3, 3), padding=(1, 1), use_bias=False)
    relu3 = sym.relu(conv3 + sym.reshape(sym.Variable(name='6', shape=(32)), shape=(1, -1, 1, 1)))
    conv4 = sym.conv2d(relu3, channels=factor**2, kernel_size=(3, 3), padding=(1, 1), use_bias=False)
    conv4 = conv4 + sym.reshape(sym.Variable(name='8', shape=(factor**2)), shape=(1, -1, 1, 1))
    # TODO(zhreshold): allow shape inference for batch size > 1
    r1 = sym.reshape(conv4, shape=(1, 1, factor, factor, size, size))
    t1 = sym.transpose(r1, axes=(0, 1, 4, 2, 5, 3))
    r2 = sym.reshape(t1, shape=(1, 1, size * factor, size * factor))
    return r2

def save_onnx_super_resolution():
    import os
    import torch
    from torch import nn
    from torch.autograd import Variable
    import torch.utils.model_zoo as model_zoo
    import torch.onnx
    import torch.nn.init as init


    class SuperResolutionNet(nn.Module):
        def __init__(self, upscale_factor, inplace=False):
            super(SuperResolutionNet, self).__init__()

            self.relu = nn.ReLU(inplace=inplace)
            self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

            self._initialize_weights()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pixel_shuffle(self.conv4(x))
            return x

        def _initialize_weights(self):
            init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
            init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
            init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
            init.orthogonal(self.conv4.weight)

    # Create the super-resolution model by using the above model definition.
    torch_model = SuperResolutionNet(upscale_factor=3)


    ######################################################################
    # Ordinarily, you would now train this model; however, for this tutorial,
    # we will instead download some pre-trained weights. Note that this model
    # was not trained fully for good accuracy and is used here for
    # demonstration purposes only.
    #

    # Load pretrained model weights
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    batch_size = 1    # just a random number

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

    # set the train mode to false since we will only run the forward pass.
    torch_model.train(False)


    ######################################################################
    # Exporting a model in PyTorch works via tracing. To export a model, you
    # call the ``torch.onnx._export()`` function. This will execute the model,
    # recording a trace of what operators are used to compute the outputs.
    # Because ``_export`` runs the model, we need provide an input tensor
    # ``x``. The values in this tensor are not important; it can be an image
    # or a random tensor as long as it is the right size.
    #
    # To learn more details about PyTorch's export interface, check out the
    # `torch.onnx documentation <http://pytorch.org/docs/master/onnx.html>`__.
    #

    # Input to the model
    x = Variable(torch.randn(batch_size, 1, 224, 224), requires_grad=True)

    # Export the model
    model_path = os.path.join(os.path.dirname(__file__), 'super_resolution.onnx')
    torch_out = torch.onnx._export(torch_model,             # model being run
                                   x,                       # model input (or a tuple for multiple inputs)
                                   model_path,              # where to save the model (can be a file or file-like object)
                                   export_params=True)      # store the trained parameter weights inside the model file
