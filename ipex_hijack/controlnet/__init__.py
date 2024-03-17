from modules.sd_hijack_utils import CondFunc
from ipex_hijack import log, hijack_message
from modules import devices


@hijack_message("Offloading annotator model to cpu")
def override_annotator_model_device(orig_func, self, *args, **kwargs):
    orig_func(self, *args, **kwargs)
    self.device = devices.cpu


def apply_controlnet_hijacks():
    def is_controlnet_device_xpu(*args, **kwargs):
        return devices.get_device_for("controlnet").type == "xpu"

    for func in [
        "annotator.manga_line.MangaLineExtration.__init__",  # lineart_anime_denoise
        "annotator.lama.LamaInpainting.__init__",  # inpaint_only+lama
        "annotator.normalbae.NormalBaeDetector.__init__", # normal_bae
        "annotator.anime_face_segment.AnimeFaceSegment.__init__", # seg_anime_face
        "annotator.oneformer.OneformerDetector.__init__", # seg_ofade20k, seg_ofcoco
    ]:
        CondFunc(
            func,
            lambda orig_func, self, *args, **kwargs: override_annotator_model_device(orig_func, self, *args, **kwargs),
            is_controlnet_device_xpu,
        )

    CondFunc(
        'annotator.uniformer.inference_segmentor', # seg_ufade20k
        lambda orig_func, model, *args, **kwargs: orig_func(model.to(devices.cpu), *args, **kwargs),
        is_controlnet_device_xpu,
    )

    log("Registered controlnet annotator hijacks")
