import re
from importlib import metadata
from pathlib import Path

from core.yolov10.yolov10_utils import emojis


def parse_version(version="0.0.0") -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        print(f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str, optional): Name to be used in warning message.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.
        msg (str, optional): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current='22.04', required='==22.04')

        # Check if current version is greater than or equal to 22.04
        check_version(current='22.10', required='22.04')  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current='22.04', required='<=22.04')

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current='21.10', required='>20.04,<22.04')
        ```
    """
    if not current:  # if current is '' or None
        print(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(emojis(f"WARNING ⚠️ {current} package is required but not installed")) from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op in (">=", "") and not (c >= v):  # if no constraint passed assume '>=required'
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(emojis(warning))  # assert version requirements met
        if verbose:
            print(warning)
    return result


def check_suffix(file="yolov8n.pt", suffix=".pt", msg=""):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"


def check_yolov5u_filename(file: str, verbose: bool = True):
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames."""
    if "yolov3" in file or "yolov5" in file:
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")  # i.e. yolov5nu.yaml -> yolov5n.yaml
        elif ".pt" in file and "u" not in file:
            original_file = file
            file = re.sub(r"(.*yolov5([nsmlx]))\.pt", "\\1u.pt", file)  # i.e. yolov5n.pt -> yolov5nu.pt
            file = re.sub(r"(.*yolov5([nsmlx])6)\.pt", "\\1u.pt", file)  # i.e. yolov5n6.pt -> yolov5n6u.pt
            file = re.sub(r"(.*yolov3(|-tiny|-spp))\.pt", "\\1u.pt", file)  # i.e. yolov3-spp.pt -> yolov3-sppu.pt
            if file != original_file and verbose:
                print(
                    f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                    f"trained with https://github.com/ultralytics/ultralytics and feature improved performance vs "
                    f"standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n"
                )
    return file
