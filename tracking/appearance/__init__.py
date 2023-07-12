import pandas as pd

def export_formats():
    # yolo tracking export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['TensorRT', 'engine', '.engine', False, True],
    ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])