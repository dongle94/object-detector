import contextlib
from pathlib import Path

import torch
import torch.nn as nn

from core.yolo.nn.modules import (
    Detect,
    v10Detect
)

try:
    import thop
except ImportError:
    thop = None

from core.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, emojis
from core.yolo.utils.checks import check_suffix
from core.yolo.utils.loss import v10DetectLoss


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale. Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # if profile:
            #     self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # if visualize:
            #     feature_visualization(x, m.type, m.i, save_dir=visualize)
            # if embed and m.i in embed:
            #     embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
            #     if m.i == max(embed):
            #         return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        # self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        #
        # # Define model
        # ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        # if nc and nc != self.yaml["nc"]:
        #     LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        #     self.yaml["nc"] = nc  # override YAML value
        # self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        # self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        # self.inplace = self.yaml.get("inplace", True)
        #
        # # Build strides
        # m = self.model[-1]  # Detect()
        # if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
        #     s = 256  # 2x min stride
        #     m.inplace = self.inplace
        #     forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)
        #     if isinstance(m, v10Detect):
        #         forward = lambda x: self.forward(x)["one2many"]
        #     m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
        #     self.stride = m.stride
        #     m.bias_init()  # only run once
        # else:
        #     self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR
        #
        # # Init weights, biases
        # initialize_weights(self)
        # if verbose:
        #     self.info()
        #     LOGGER.info("")


class YOLOv10DetectionModel(DetectionModel):
    def init_criterion(self):
        return v10DetectLoss(self)


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


@contextlib.contextmanager
def temporary_modules(modules=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.

    Example:
        ```python
        with temporary_modules({'old.module.path': 'new.module.path'}):
            import old.module.path  # this will now import new.module.path
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if not modules:
        modules = {}

    import importlib
    import sys

    try:
        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = importlib.import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):
    from core.yolo.utils.downloads import attempt_download_asset
    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            {
                "ultralytics.yolo.utils": "core.yolov10.yolov10_utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
                'ultralytics.nn.tasks': 'core.yolov10.nn.tasks',
                'ultralytics.nn.modules.block': 'core.yolov10.nn.modules.block',
                'ultralytics.nn.modules.head': 'core.yolov10.nn.modules.head',
            }
        ):  # for legacy 8.0 Classify and Pose models
            ckpt = torch.load(file, map_location="cpu")
    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        print(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        # check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")
        exit()

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        print(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file  # load


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(
            model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt



def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if m == "detect" or m == "v10detect":
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            # if isinstance(m, Segment):
            #     return "segment"
            # elif isinstance(m, Classify):
            #     return "classify"
            # elif isinstance(m, Pose):
            #     return "pose"
            # elif isinstance(m, OBB):
            #     return "obb"
            if isinstance(m, Detect, v10Detect): #, WorldDetect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        # if "-seg" in model.stem or "segment" in model.parts:
        #     return "segment"
        # elif "-cls" in model.stem or "classify" in model.parts:
        #     return "classify"
        # elif "-pose" in model.stem or "pose" in model.parts:
        #     return "pose"
        # elif "-obb" in model.stem or "obb" in model.parts:
        #     return "obb"
        if "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    print(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
