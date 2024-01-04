from modules.sd_hijack_utils import CondFunc
from ipex_hijack import log, hijack_message
from modules import devices


# Adapted from https://github.com/Mikubill/sd-webui-controlnet/blob/4cf15d1c9c565b8d0c5f782a89c5a6286dc6e6ff/annotator/leres/leres/depthmap.py#L34
@hijack_message("Offloading estimateleres() to cpu")
def estimateleres_cpu(img, model, w, h):
    from annotator.leres.leres.depthmap import torch, cv2, scale_torch

    # leres transform input
    rgb_c = img[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (w, h))
    img_torch = scale_torch(A_resize)[None, :, :, :]

    # compute
    model.to(devices.cpu)
    with torch.no_grad():
        img_torch = img_torch.to(devices.cpu)
        prediction = model.depth_model(img_torch)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(
        prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC
    )

    return prediction


@hijack_message("Offloading annotator model to cpu")
def override_annotator_model_device(orig_func, self, *args, **kwargs):
    orig_func(self, *args, **kwargs)
    self.device = devices.cpu


# Adapted from https://github.com/Mikubill/sd-webui-controlnet/blob/4cf15d1c9c565b8d0c5f782a89c5a6286dc6e6ff/annotator/mlsd/utils.py#L48
@hijack_message("Offloading pred_lines() to cpu")
def pred_lines_cpu(image, model,
               input_shape=[512, 512],
               score_thr=0.10,
               dist_thr=20.0):
    from annotator.mlsd.utils import np, cv2, torch, deccode_output_score_and_ptss
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

    resized_image = np.concatenate([cv2.resize(image, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA),
                                    np.ones([input_shape[0], input_shape[1], 1])], axis=-1)

    resized_image = resized_image.transpose((2,0,1))
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    batch_image = (batch_image / 127.5) - 1.0

    batch_image = torch.from_numpy(batch_image).float().to(devices.cpu)
    model = model.to(devices.cpu)
    outputs = model(batch_image).to(devices.cpu)
    pts, pts_score, vmap = deccode_output_score_and_ptss(outputs, 200, 3)
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    lines = 2 * np.array(segments_list)  # 256 > 512
    lines[:, 0] = lines[:, 0] * w_ratio
    lines[:, 1] = lines[:, 1] * h_ratio
    lines[:, 2] = lines[:, 2] * w_ratio
    lines[:, 3] = lines[:, 3] * h_ratio

    return lines


def apply_controlnet_hijacks():
    def is_controlnet_device_xpu(*args, **kwargs):
        return devices.get_device_for("controlnet").type == "xpu"

    for func in [
        "annotator.leres.leres.depthmap.estimateleres",  # depth_leres++
        "annotator.leres.estimateleres",  # depth_leres
    ]:
        CondFunc(
            func,
            lambda _, *args, **kwargs: estimateleres_cpu(*args, **kwargs),
            is_controlnet_device_xpu,
        )

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
        'annotator.mlsd.pred_lines', # mlsd
        lambda _, *args, **kwargs: pred_lines_cpu(*args, **kwargs),
        is_controlnet_device_xpu,
    )

    CondFunc(
        'annotator.uniformer.inference_segmentor', # seg_ufade20k
        lambda orig_func, model, *args, **kwargs: orig_func(model.to(devices.cpu), *args, **kwargs),
        is_controlnet_device_xpu,
    )

    log("Registered controlnet annotator hijacks")
