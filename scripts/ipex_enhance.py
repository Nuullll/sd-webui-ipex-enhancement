from modules import script_callbacks, devices
from modules.sd_hijack_utils import CondFunc
from ipex_hijack import log
from ipex_hijack.controlnet import apply_controlnet_hijacks


if devices.has_xpu():
    import torch
    import intel_extension_for_pytorch as ipex


def ipex_optimize(sd_model):
    if sd_model.device.type == "xpu":
        try:
            ipex.optimize(
                sd_model,
                dtype=devices.dtype_unet,
                inplace=True,
                # conv_bn_folding=False,
                linear_bn_folding=True,
                # weights_prepack=False,
                # graph_mode=True,
            )
            log("Applied IPEX optimize.")
        except Exception:
            log("Warning: couldn't apply IPEX optimize because part of SD model is mapped to CPU")


def apply_general_hijacks():
    CondFunc('torchvision.ops.nms',
        lambda orig_func, boxes, scores, iou_threshold: orig_func(boxes.to(devices.get_optimal_device()), scores.to(devices.get_optimal_device()), iou_threshold).to(boxes.device),
        lambda orig_func, boxes, scores, iou_threshold: not boxes.is_xpu or not scores.is_xpu)

    # IPEX: incorrect batch_norm result with XPU fp32, downcast to fp16 instead
    # TODO: file an issue to IPEX
    CondFunc('torch.nn.functional.batch_norm',
        lambda orig_func, input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05: orig_func(input.half(), running_mean.half(), running_var.half(), weight=weight.half() if weight is not None else None, bias=bias.half() if bias is not None else None, training=training, momentum=momentum, eps=eps).to(input.dtype),
        lambda orig_func, input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05: input.device.type == 'xpu' and input.dtype == torch.float)

    log("Registered hijacks for IPEX")


if devices.has_xpu():
    script_callbacks.on_model_loaded(ipex_optimize)
    log("Registered IPEX model optimize callback")

    apply_general_hijacks()
    apply_controlnet_hijacks()
