import os
import platform
import torch

from core.yolov10 import __version__
from core.yolov10.yolov10_utils.checks import check_version


TORCH_2_0 = check_version(torch.__version__, "2.0.0")


# def get_cpu_info():
#     """Return a string with system CPU information, i.e. 'Apple M2'."""
#     import cpuinfo  # pip install py-cpuinfo
#
#     k = "brand_raw", "hardware_raw", "arch_string_raw"  # info keys sorted by preference (not all keys always available)
#     info = cpuinfo.get_cpu_info()  # info dict
#     string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], "unknown")
#     return string.replace("(R)", "").replace("CPU ", "").replace("@ ", "")


def select_device(device="", gpu_num=0, newline=False, verbose=True):
    """
    Selects the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object.
            Options are 'None', 'cpu', or 'cuda', or '0' or '0,1,2,3'. Defaults to an empty string, which auto-selects
            the first available GPU, or CPU if no GPU is available.
        gpu_num: (int, optional): Number of GPU devices to use. Defaults to 0
        newline (bool, optional): If True, adds a newline at the end of the log string. Defaults to False.
        verbose (bool, optional): If True, logs the device information. Defaults to True.

    Returns:
        (torch.device): Selected device.

    Raises:
        ValueError: If the specified device is not available or if the batch size is not a multiple of the number of
            devices when using multiple GPUs.

    Examples:
        >>> select_device('cuda:0')
        device(type='cuda', index=0)

        >>> select_device('cpu')
        device(type='cpu')

    Note:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """

    if isinstance(device, torch.device):
        return device

    s = f"Ultralytics YOLOv{__version__} 🚀 Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    mps = device in ("mps", "mps:0")  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        # if device == "cuda":
        #     device = "0"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        # os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            print(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        # devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        # n = len(devices)  # device count
        # if n > 1 and batch > 0 and batch % n != 0:  # check batch_size is divisible by device_count
        #     raise ValueError(
        #         f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
        #         f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}."
        #     )
        # space = " " * (len(s) + 1)
        # for i, d in enumerate(devices):
        #     p = torch.cuda.get_device_properties(i)
        #     s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = f"cuda:{gpu_num}"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # Prefer MPS if available
        # s += f"MPS ({get_cpu_info()})\n"
        arg = "mps"
    else:  # revert to CPU
        # s += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"

    if verbose:
        print(s if newline else s.rstrip())
    return torch.device(arg)
