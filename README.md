# sd-webui-ipex-enhancement

This is an extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), to optimize user experience with IPEX backend.

## Features

- Apply [`ipex.optimize`](https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/api_doc.html) upon model loading. There is a **~50%** chance that you will get a **~10%** performance gain (`ipex.optimize` seems not deterministic).
- Offload the following **controlnet annotators** to CPU, as IPEX XPU backend doesn't work as expected.
  - depth_leres, depth_leres++
  - lineart_anime_denoise
  - inpaint_only+lama
  - normal_bae
  - mlsd
  - seg_anime_face, seg_ofade20k, seg_ofcoco, seg_ufade20k
- Offload `torchvision.ops.nms` to XPU, to W/A flaky Access Violation when using [adetailer](https://github.com/Bing-su/adetailer) with `torchvision==0.15.2a0+fa99a53`.
