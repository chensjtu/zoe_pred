import torch
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import get_image_from_url, colorize

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from pprint import pprint
import os 
os.environ['http_proxy'] = "http://127.0.0.1:15777"
os.environ['https_proxy'] = "http://127.0.0.1:15777"
from zoedepth.utils.misc import colorize


def save_raw_depth(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 1000  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)

def load_raw_depth(fpath="raw.png"):
    depth = Image.open(fpath)
    depth = np.array(depth)
    depth = (depth / 1000).astype(np.float32)
    return depth


if __name__ == "__main__":
    # first, we check the save and load functions
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)

    # model = torch.hub.load(".", "ZoeD_K", source="local", pretrained=True) # pretrained on kitti
    model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True) # pretrained on nyu
    # model = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True) # pretrained on nyu and kitti

    # in this case, we are using a pretrained model, so we need to specify the input size
    ##### sample prediction
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)


    # Local file
    from PIL import Image
    image = Image.open("sparse_nerf_datasets/sparse_omni3d_undistorted/backpack_016/images/00000.jpg").convert("RGB")  # load
    depth_numpy = zoe.infer_pil(image)  # as numpy
    depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image
    depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

    fpath = "./out.png"
    save_raw_depth(depth_numpy, fpath)
    # # Tensor 
    # from zoedepth.utils.misc import pil_to_batched_tensor
    # X = pil_to_batched_tensor(image).to(DEVICE)
    # depth_tensor = zoe.infer(X)

    # # Save raw
    # from zoedepth.utils.misc import save_raw_16bit
    # fpath = "./out.png"
    # save_raw_16bit(depth_numpy, fpath)

    # Colorize output
    colored = colorize(depth_numpy, cmap="magma_r")

    # save colored output
    fpath_colored = "./output_colored.png"
    Image.fromarray(colored).save(fpath_colored)

    # # Load raw
    depth_numpy_load = load_raw_depth(fpath)
    # just check that it is the same
    print(np.allclose(depth_numpy, depth_numpy_load))
    # if not same, the error should be very small
    print(np.abs(depth_numpy - depth_numpy_load).max())


