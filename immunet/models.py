import os
from typing import Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from models_parts import *
import torch.nn as nn


@staticmethod
def _print_network(self, _print=True):
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/14422fb8486a4a2bd991082c1cda50c3a41a755e/models/networks.py
    if isinstance(self, list):
        self = self[0]
    num_params = 0
    for param in self.parameters():
        num_params += param.numel()
    _str = (
        "Network [%s] was created. Total number of parameters: %.1f million. "
        "To see the architecture, do print(network)."
        % (type(self).__name__, num_params / 1000000)
    )
    if _print:
        print(_str)
    return _str


class BaseImmunet(nn.Module):
    """Base class, expects N, C, H, W"""

    def __init__(
        self,
        n_inputs: int,
        n_markers: int,
        n_rays: int,
        window_size: int,
        flatten_flag: bool,
        debug_flag: bool,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_markers = n_markers
        self.n_rays = n_rays
        self.window = window_size
        self.flatten = flatten_flag
        self.debug = debug_flag

        ## Calulating output
        ## nh x nw (input height x width)
        ## kh x kw (kernel height x width)
        ## output = (nh - kh + 1, nw - kw + 1)

    def forward(self, inputs):
        if self.debug:
            # make sure format is N, C, H, W
            assert (
                inputs.size(1) == self.n_inputs
            ), f"Channels should come after batch dim!, {inputs.size()}"
            # assert inputs.size(2) >= inputs.size(3), "Height should come before width!"

        if self.training:
            a = inputs
        else:
            w = self.window
            # starting from last dimension
            # pad last dim by w,w
            # pad second to last by w, w
            # rest untouched
            pad = (w, w, w, w)
            a = F.pad(inputs, pad, mode="reflect")

        return a

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        self.toggle_gradients_distance(False)
        self.toggle_gradients_phenotype(False)
        self.toggle_gradients_rays(False)

    def toggle_gradients_distance(self, freeze: bool):
        for param in self.distance.parameters():
            param.requires_grad = not freeze

    def toggle_gradients_phenotype(self, freeze: bool):
        for param in self.phenotype.parameters():
            param.requires_grad = not freeze

    def toggle_gradients_rays(self, freeze: bool):
        for param in self.rays.parameters():
            param.requires_grad = not freeze

    def print_network(self, _print=True):
        return _print_network(self, _print)

    @classmethod
    def from_path(
        cls,
        backbone: str,
        model_weights_path: str | os.PathLike,
        debug: bool = False,
        device: Optional[str | torch.device] = None,
    ) -> "BaseImmunet":

        weight_dict = torch.load(model_weights_path, weights_only=True)
        out_markers_num = list(weight_dict.values())[-2].shape[0]
        input_num = list(weight_dict.values())[0].shape[1]

        model: BaseImmunet
        if backbone == "ORIGINAL":
            model = ImmuNet(
                n_inputs=input_num, n_markers=out_markers_num, debug_flag=debug
            )

        elif backbone == "UNET":
            num_features = weight_dict["features.weight"].shape[0]
            base_channels = list(weight_dict.values())[0].shape[0]
            model = UImmNet(
                n_inputs=input_num,
                n_markers=out_markers_num,
                base_channels=base_channels,
                num_features=num_features,
                debug_flag=debug,
            )

        elif backbone == "DAPI":
            base_channels = list(weight_dict.values())[0].shape[0]
            num_features = weight_dict["features.weight"].shape[0]
            model = DAPImmuNet(
                base_channels=base_channels, num_features=num_features, debug_flag=debug
            )

        else:
            raise NotImplementedError(f"{backbone} not implemented or misspelled!")

        try:
            model.load_state_dict(weight_dict)
        except RuntimeError as E:
            raise Exception(
                f"couldn't load model, make sure you are using the right backbone «{backbone}»"
            ) from E

        if device:
            model.to(device)

        return model


class ImmuNet(BaseImmunet):
    def __init__(
        self,
        n_inputs=7,
        n_markers=5,
        n_rays=0,
        window_size=30,
        flatten_flag=False,
        debug_flag=False,
    ) -> None:
        super().__init__(
            n_inputs, n_markers, n_rays, window_size, flatten_flag, debug_flag
        )
        # block 1
        input_block_1 = n_inputs
        output_block_1 = 64
        self.main_1 = ImmuNetMainBlock(
            input_block_1,
            output_block_1,
            kernel_sizes=[4, 3],
            strides=[1, 1],
            dilations=[1, 1],
            block_index=1,
        )
        self.skip_connection_1 = ImmuNetSkipBlock(
            input_block_1, output_block_1, kernel_size=6, dilation=1, block_index=1
        )
        self.end_1 = ImmuNetEndBlock(
            kernel_size=2, stride=1, dilation=1, padding=0, block_index=1
        )

        # block 2
        input_block_2 = output_block_1
        output_block_2 = 128
        self.main_2 = ImmuNetMainBlock(
            input_block_2,
            output_block_2,
            kernel_sizes=[3, 3],
            strides=[1, 1],
            dilations=[2, 2],
            block_index=3,
        )
        self.skip_connection_2 = ImmuNetSkipBlock(
            input_block_2, output_block_2, kernel_size=5, dilation=2, block_index=2
        )
        self.end_2 = ImmuNetEndBlock(
            kernel_size=2, stride=1, dilation=2, padding=0, block_index=2
        )

        # block 3
        input_block_3 = output_block_2
        output_block_3 = 256
        self.main_3 = ImmuNetMainBlock(
            input_block_3,
            output_block_3,
            kernel_sizes=[3, 3],
            strides=[1, 1],
            dilations=[4, 4],
            block_index=5,
        )
        self.skip_connection_3 = ImmuNetSkipBlock(
            input_block_3, output_block_3, kernel_size=5, dilation=4, block_index=3
        )
        self.end_3 = ImmuNetEndBlock(kernel_size=2, stride=1, dilation=4, block_index=3)

        # block 4
        input_block_4 = output_block_3
        output_block_4 = 512
        self.main_4 = ImmuNetMainBlock(
            input_block_4,
            output_block_4,
            kernel_sizes=[4, 1],
            strides=[1, 1],
            dilations=[8, 1],
            final_batch_norm=False,
            block_index=7,
        )

        # distance branch
        self.distance = ImmuNetOutputBranch(
            n_features=output_block_4,
            n_outputs=1,
            dropout=True,
            layer_index=9,
            kernel_size=1,
            stride=1,
        )

        # phenotype branch
        self.phenotype = ImmuNetOutputBranch(
            n_features=output_block_4,
            n_outputs=self.n_markers,
            dropout=True,
            layer_index=15,
            kernel_size=1,
            stride=1,
        )

        ## TODO: enable RAYS
        # self.rays = ImmuNetOutputBranch(
        #     n_features=output_block_4,
        #     n_outputs=self.n_rays,
        #     dropout=True,
        #     layer_index=13,
        #     kernel_size=1,
        #     stride=1,
        # )

    @classmethod
    def from_keras_weights(cls, keras_weights: dict):
        """Create ImmuNet from keras weights into PyTorch model.

        Usage:
            `pytorch_model = ImmuNet.from_keras_weights(keras_weights)`

        Parameters
        ----------
        keras_weights : dict
            Dictionary containing weights, where the key is the name of the layer and
            the value are the weights/bias themselves (using `get_weights()` method) Example
            usage:

            `keras_weights = {}`

            `for layer in keras_model.layers:`
                `layer_name = layer.name`

                `keras_weights[layer_name] = layer.get_weights()`

        Returns
        -------
        ImmuNet
            Returns model with pretrained weights transfered.

        Raises
        ------
        Exception
            _description_
        E
            _description_
        """
        weight_index = 0
        first_conv_name = "conv1"
        last_pheno_conv_name = "conv16"
        assert (
            last_pheno_conv_name in keras_weights
        ), "Model is missing phenotype output, behaviour not implemented " + str(
            keras_weights.keys()
        )

        input_num_keras = keras_weights[first_conv_name][weight_index].shape[2]
        out_markers_num_keras = keras_weights[last_pheno_conv_name][weight_index].shape[
            -1
        ]

        instance: ImmuNet = cls(
            n_inputs=input_num_keras,
            n_markers=out_markers_num_keras,
            n_rays=0,
            window_size=30,
            flatten_flag=False,
            debug_flag=True,
        )

        # assert keras_weights[first_conv_name][weight_index].shape[2] == 7, keras_weights[first_conv_name][weight_index].shape[2]
        import copy

        def models_equal(model_1, model_2):
            models_differ = 0
            for key_item_1, key_item_2 in zip(
                model_1.state_dict().items(), model_2.state_dict().items()
            ):
                if torch.equal(key_item_1[1], key_item_2[1]):
                    pass
                else:
                    models_differ += 1
                    if key_item_1[0] == key_item_2[0]:
                        pass
                    else:
                        raise RuntimeError("Couldn't change model parameter!")
            if models_differ == 0:
                print("Models match perfectly!")
                return True
            return False

        # backup model
        pt_model = copy.deepcopy(instance)

        state_dict = instance.state_dict()
        for name, py_layer in state_dict.items():
            with torch.no_grad():
                try:
                    layer_name = name.split(".")[-2]
                    if layer_name not in keras_weights:
                        continue
                    weights = keras_weights[layer_name]
                    weight_type = name.split(".")[-1]

                    if "weight" == weight_type:
                        if len(weights[0].shape) == 2:
                            # Keras Linear Layer: input neurons * output neurons
                            # PyTorch Linear Layer: output neurons * input neurons
                            py_layer.copy_(torch.from_numpy(weights[0]).T)
                        elif len(weights[0].shape) == 4:
                            # Keras 2D Convolutional layer: height * width * input channels * output channels
                            # PyTorch 2D Convolutional layer: output channels * input channels * height * width
                            py_layer.copy_(
                                torch.from_numpy(weights[0]).permute(3, 2, 0, 1)
                            )
                        else:
                            py_layer.copy_(torch.from_numpy(weights[0]))

                    elif "bias" == weight_type:
                        py_layer.copy_(torch.from_numpy(weights[1]))

                    elif "running_mean" == weight_type:
                        py_layer.copy_(torch.from_numpy(weights[2]))

                    elif "running_var" == weight_type:
                        py_layer.copy_(torch.from_numpy(weights[3]))
                    else:
                        continue

                except Exception as E:
                    print("EXCEPTION")
                    print(name.split(".")[-1], name, name.split(".")[-2])
                    print("Weights vs py layer shape", weights[0].shape, py_layer.shape)
                    raise E

        instance.load_state_dict(state_dict)

        input_num_pytorch = list(state_dict.values())[0].shape[1]
        out_markers_num_pytorch = list(state_dict.values())[-2].shape[0]

        assert input_num_pytorch == input_num_keras
        assert out_markers_num_pytorch == out_markers_num_keras

        # make sure changes took effect
        assert models_equal(instance, pt_model) == False
        del pt_model

        instance.debug = False

        return instance

    def forward(self, inputs):
        a = super().forward(inputs)

        # block 1
        x = self.main_1(a)
        a = self.skip_connection_1(a)
        x = torch.add(x, a)
        a = self.end_1(x)

        # block 2
        x = self.main_2(a)
        a = self.skip_connection_2(a)
        x = torch.add(x, a)
        a = self.end_2(x)

        # block 3
        x = self.main_3(a)
        a = self.skip_connection_3(a)
        x = torch.add(x, a)
        x = self.end_3(x)

        # block 4
        dist_intermediate = self.main_4(x)
        del x, a
        distance = self.distance(dist_intermediate)

        if self.n_markers > 0:
            phenotype = self.phenotype(dist_intermediate)
            return distance, phenotype

        return distance, torch.zeros(1, device=inputs.device)
        ## TODO: enable RAYS
        # rays = torch.zeros(1, device=inputs.device)
        # if self.n_rays > 0:
        #     rays = self.rays(dist_intermediate)

        # do not use list for better trace

        # return distance, phenotype, rays


class UImmNet(BaseImmunet):

    def __init__(
        self,
        n_inputs=7,
        n_markers=5,
        n_rays=0,
        base_channels=32,
        num_features=256,
        window_size=0,
        flatten_flag=False,
        debug_flag=False,
        bilinear=True,
    ) -> None:
        super().__init__(
            n_inputs, n_markers, n_rays, window_size, flatten_flag, debug_flag
        )

        self.inc = DoubleConv(self.n_inputs, base_channels)
        factor = 2 if bilinear else 1

        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8 // factor)

        self.up1 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up2 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up3 = Up(base_channels * 2, base_channels, bilinear)

        self.num_features = num_features
        self.features = nn.Conv2d(base_channels, self.num_features, 3, padding=1)

        # distance branch
        self.distance = ImmuNetOutputBranch(
            n_features=self.num_features,
            n_outputs=1,
            dropout=True,
            layer_index=9,
            kernel_size=1,
            stride=1,
        )

        # phenotype branch
        self.phenotype = ImmuNetOutputBranch(
            n_features=self.num_features,
            n_outputs=self.n_markers,
            dropout=True,
            layer_index=15,
            kernel_size=1,
            stride=1,
        )

        ## TODO: enable RAYS
        # self.rays = ImmuNetOutputBranch(
        #     n_features=self.num_features,
        #     n_outputs=self.n_rays,
        #     dropout=True,
        #     layer_index=13,
        #     kernel_size=1,
        #     stride=1,
        # )

    def forward(self, inputs):
        a = super().forward(inputs)

        # U-net
        x1 = self.inc(a)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # combines last two downConvs
        x = self.up1(x4, x3)

        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # After the final U-Net feature layer, we cautiously add an additional
        # 3×3 convolutional layer with 128 channels (and relu activations)
        # to avoid that the subsequent two output layers have to “fight over features”.
        x = self.features(x)

        # Specifically, we use a single-channel convolutional layer with sigmoid activation
        # for the object probability output.
        distance = self.distance(x)

        # Phenotype output layer has as many channels as number of markers.
        phenotype = self.phenotype(x)

        ## TODO: enable RAYS
        # rays = torch.zeros(1, device=inputs.device)
        # if self.rays:
        #     # The polygon distance output layer has as many channels as there are radial
        #     # directions n and does not use an additional activation function.
        #     rays = self.rays(x)

        if self.flatten:
            distance = torch.flatten(distance, start_dim=1)
            phenotype = torch.flatten(phenotype, start_dim=1)
            # if self.rays:
            #     rays = torch.flatten(rays, start_dim=1)

        return distance, phenotype
        # return distance, phenotype, rays


def load_model(
    device: str,
    which_model: Union[str, os.PathLike],
    debug=False,
    backbone: str = "ORIGINAL",
) -> nn.Module:
    """Loads model from path, and moves it to assigned device.

    Parameters
    ----------
    device : str,
        Device (cuda, cpu, etc) to move the model to.
    which_model : Union[str, Path]
        Path to load model.
    debug : bool, optional
        Flag to enable verbose output, by default False
    backbone : str, optional
        Specify architecture of the model, by default ORIGINAL

    Returns
    -------
    torch.nn.Module
        _description_
    """
    model_path = which_model
    if ".pth" not in str(which_model):
        model_path = f"{which_model}.pth"
    model = BaseImmunet.from_path(
        backbone=backbone, model_weights_path=model_path, debug=debug, device=device
    )

    # model.to(device)
    model.eval()
    return model


class DAPImmuNet(BaseImmunet):
    def __init__(
        self,
        debug_flag=False,
        n_inputs=1,
        num_features=32,
        base_channels=32,
        bilinear=True,
    ):
        super().__init__(
            n_inputs=n_inputs,
            n_markers=0,
            n_rays=0,
            window_size=0,
            flatten_flag=False,
            debug_flag=debug_flag,
        )

        self.debug_flag = debug_flag
        self.n_inputs = n_inputs  # TODO: two if we also use the AF channel

        self.inc = DoubleConv(self.n_inputs, base_channels)
        factor = 2 if bilinear else 1

        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8 // factor)

        self.up1 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up2 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up3 = Up(base_channels * 2, base_channels, bilinear)

        self.num_features = num_features
        self.features = nn.Conv2d(base_channels, self.num_features, 3, padding=1)

        self.dapi = OutConv(
            in_channels=self.num_features,
            out_channels=1,
        )

    def forward(self, inputs):

        # U-net
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # combines last two downConvs
        x = self.up1(x4, x3)

        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # After the final U-Net feature layer, we cautiously add an additional
        # convolutional layer (and relu activations) to avoid that the subsequent
        # two output layers have to “fight over features”.
        x = self.features(x)

        # Specifically, we use a single-channel convolutional layer
        # raw logits
        dapi = self.dapi(x)

        return dapi, torch.zeros(1, device=inputs.device)


# TODO:
# class ResImmuNet(BaseImmunet):
#     def __init__(
#         self,
#         n_inputs=7,
#         n_markers=5,
#         n_rays=8,
#         window_size=32,
#         flatten_flag=False,
#         debug_flag=False,
#     ):
#         from torchvision.models.resnet import BasicBlock, ResNet

#         super().__init__(
#             n_inputs, n_markers, n_rays, window_size, flatten_flag, debug_flag
#         )

#         num_features = 0

#         # Create ResNet, modify first convolution
#         # Reduces the input height and width to 1/32 of the original
#         _resNet = ResNet(BasicBlock, [2, 2, 2, 2])
#         # "model surgery"
#         _resNet.conv1 = nn.Conv2d(
#             self.n_inputs,
#             out_channels=64,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False,
#         )

#         # copies all layers in the ResNet-18 except for the final global
#         # average pooling layer and the fully connected layer that are closest to the output.
#         self.resnet = nn.Sequential(*list(_resNet.children())[:-2])
#         num_features = 512

#         self.build_output(num_features, n_rays)

#     def forward(self, inputs):
#         inputs = super().forward(inputs)

#         x = self.resnet(inputs)

#         # Specifically, we use a single-channel convolutional layer with sigmoid activation
#         # for the object probability output.
#         distance = self.distance(x)

#         rays = torch.zeros(1, device=inputs.device)
#         if self.rays:
#             # The polygon distance output layer has as many channels as there are radial
#             # directions n and does not use an additional activation function.
#             rays = self.rays(x)

#         # Phenotype output layer has as many channels as number of markers.
#         phenotype = self.phenotype(x)

#         if self.training or self.flatten:
#             distance = torch.flatten(distance, start_dim=1)
#             phenotype = torch.flatten(phenotype, start_dim=1)
#             if self.rays:
#                 rays = torch.flatten(rays, start_dim=1)

#         return distance, phenotype, rays
