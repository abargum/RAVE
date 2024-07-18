import torch

torch.set_grad_enabled(False)
import os

import cached_conv as cc
import gin
import torch.nn as nn
from absl import app, flags
from effortless_config import Config

import rave

flags.DEFINE_string('run', default=None, required=True, help='Run to export')
FLAGS = flags.FLAGS


def main(argv):
    gin.parse_config_file(os.path.join(FLAGS.run, "config.gin"))
    checkpoint = rave.core.search_for_run(FLAGS.run)

    print(f"using {checkpoint}")

    pretrained = rave.RAVE()
    pretrained.load_state_dict(torch.load(checkpoint)["state_dict"])
    pretrained.eval()

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

    def recursive_replace(model: nn.Module):
        for name, child in model.named_children():
            if isinstance(child, cc.convs.Conv1d):
                padding = child._pad[0] if isinstance(child._pad, (list, tuple)) else child._pad
                
                conv = nn.Conv1d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    padding,  # Proper padding value
                    child.dilation,
                    child.groups,
                    child.bias is not None,  # Ensuring bias is a boolean value
                )
                conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    conv.bias.data.copy_(child.bias.data)
                setattr(model, name, conv)
            elif isinstance(child, cc.convs.ConvTranspose1d):
                padding = child.padding[0] if isinstance(child.padding, (list, tuple)) else child.padding
                
                conv = nn.ConvTranspose1d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    padding,  # Proper padding value
                    child.output_padding,
                    child.groups,
                    child.bias is not None,  # Ensuring bias is a boolean value
                    child.dilation,
                    child.padding_mode,
                )
                conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    conv.bias.data.copy_(child.bias.data)
                setattr(model, name, conv)
            else:
                recursive_replace(child)

    recursive_replace(pretrained)

    x = torch.randn(1, 1, 2**15)
    pretrained(x)

    name = os.path.basename(os.path.normpath(FLAGS.run))
    export_path = os.path.join(FLAGS.run, name)
    torch.onnx.export(
        pretrained,
        x,
        f"{export_path}.onnx",
        export_params=True,
        opset_version=17,
        input_names=["audio_in"],
        output_names=["audio_out"],
        dynamic_axes={
            "audio_in": {
                2: "audio_length"
            },
            "audio_out": [0],
        },
        do_constant_folding=False,
    )


if __name__ == '__main__':
    app.run(main)