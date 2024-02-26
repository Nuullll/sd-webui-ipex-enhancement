from modules import script_callbacks, devices
from modules.sd_hijack_utils import CondFunc
from ipex_hijack import log
from ipex_hijack.controlnet import apply_controlnet_hijacks


if devices.has_xpu():
    import torch
    import intel_extension_for_pytorch as ipex


def ipex_optimize(sd_model):
    # SD WebUI runs model_loaded_callback before moving to target device.
    # W/A: Move to xpu before applying ipex.optimize
    # TODO: Fix SD WebUI
    if not sd_model.lowvram:
        sd_model.to(devices.device)

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

    log("Registered hijacks for IPEX")


if devices.has_xpu():
    script_callbacks.on_model_loaded(ipex_optimize)
    log("Registered IPEX model optimize callback")

    apply_general_hijacks()
    apply_controlnet_hijacks()
