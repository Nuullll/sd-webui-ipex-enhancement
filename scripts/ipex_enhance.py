from modules import script_callbacks, devices
from modules.sd_hijack_utils import CondFunc
from ipex_hijack import log, asfp16, asfp32
from ipex_hijack.controlnet import apply_controlnet_hijacks
from pkg_resources import parse_version


if devices.has_xpu():
    import torch
    import intel_extension_for_pytorch as ipex
    ipex_ver = parse_version(ipex.__version__)
    log(f"Using IPEX version: {ipex_ver}")


def ipex_optimize(sd_model):
    if sd_model.device.type == "xpu":
        try:
            ipex.optimize(
                sd_model,
                dtype=devices.dtype_unet,
                inplace=True,
                # conv_bn_folding=False,
                # linear_bn_folding=True,
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
    # The issue has been fixed since IPEX 2.1.30
    if ipex_ver < parse_version('2.1.30'):
        CondFunc('torch.nn.functional.batch_norm',
            lambda orig_func, input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05: orig_func(input.half(), asfp16(running_mean), asfp16(running_var), weight=asfp16(weight), bias=asfp16(bias), training=training, momentum=momentum, eps=eps).to(input.dtype),
            lambda orig_func, input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05: input.device.type == 'xpu' and input.dtype == torch.float)
    
    # IPEX: incorrect interpolate result with XPU when align_corners=True, move to cpu instead
    # The issue has been fixed since IPEX 2.1.30
    if ipex_ver < parse_version('2.1.30'):
        CondFunc('torch.nn.functional.interpolate',
            lambda orig_func, input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False: orig_func(input.cpu(), size, scale_factor, mode, align_corners, recompute_scale_factor, antialias).to(input.device),
            lambda orig_func, input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False: input.device.type == 'xpu' and align_corners)

    # Fixes ipadapter on CPU
    CondFunc('torch.nn.functional.linear',
        lambda orig_func, input, weight, bias: orig_func(input.float(), asfp32(weight), asfp32(bias)).half(),
        lambda orig_func, input, weight, bias: input.device.type == 'cpu' and input.dtype == torch.half)

    if ipex_ver >= parse_version('2.5'):
        # disable sdpa workaround for IPEX >= 2.5
        log("IPEX version >= 2.5, disable sdpa workaround")
        import modules.xpu_specific
        modules.xpu_specific.torch_xpu_scaled_dot_product_attention = modules.xpu_specific.orig_sdp_attn_func

    log("Registered hijacks for IPEX")


if devices.has_xpu():
    script_callbacks.on_model_loaded(ipex_optimize)
    log("Registered IPEX model optimize callback")

    apply_general_hijacks()
    apply_controlnet_hijacks()
