# sd-webui-ipex-enhancement

This is an extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), to optimize user experience with IPEX backend.

## Features

- Apply [`ipex.optimize`](https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/api_doc.html) upon model loading.
- Fix `torch.nn.functional.batch_norm` for XPU fp32, IPEX < 2.1.30
  + Known impact: controlnet annotator depth_leres, depth_leres++
- Fix `torch.nn.functional.interpolate` for XPU when `align_corners=True`, IPEX < 2.1.30
  + Known impact: controlnet annotator MLSD
- Offload the following **controlnet annotators** to CPU, as IPEX XPU backend doesn't work as expected.
  - lineart_anime_denoise
  - inpaint_only+lama
  - normal_bae
  - mlsd
  - seg_anime_face, seg_ofade20k, seg_ofcoco, seg_ufade20k
- Offload `torchvision.ops.nms` to XPU, to W/A flaky Access Violation when using [adetailer](https://github.com/Bing-su/adetailer) with `torchvision==0.15.2a0+fa99a53`.
