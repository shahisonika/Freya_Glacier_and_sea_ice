#!/usr/bin/env python3

import argparse
import sys
import argcomplete
import inspect
import ast
# from FG0_preprocess_data import SVDDataPreprocessor, CreateRacmoMasks, ExtractFreyaGlacierVariablesFromRacmo
from FG2_mca_plots import SingularVectors, HeterogeneousCorrelation

# -------------------------
# Import the MCA modules
# -------------------------
# from FG1_mca import CalculateMCA
from mca_qr import QRDecomposition
from mca_rotqr import RotQRDecomposition
from mca_rotxeofs import RotXeofs
# from mca_xeofs import Xeofss
# from mca_xeofs_revised import Xeofs
from mca_xeofs_weekly import Xeofs
from complex_mca import ComplexRotatedMCA

def get_class_methods(class_obj):
    """Return callable method names from a class (excluding __dunder__)."""
    return [
        func for func in dir(class_obj)
        if callable(getattr(class_obj, func)) and not func.startswith("__")
    ]

def get_method_description(class_obj, method_name):
    """Get first line of method's docstring or default text."""
    method = getattr(class_obj, method_name, None)
    if not callable(method):
        return "Not a callable method"
    doc = method.__doc__
    if doc:
        return doc.strip().split("\n")[0]
    return "No description available"

def get_method_parameters(cls, method_name):
    method = getattr(cls, method_name, None)
    if not callable(method):
        return []
    sig = inspect.signature(method)
    # skip self parameter
    return [p.name for p in sig.parameters.values() if p.name != 'self']

class_map = {
    # "svd_data_preprocess": SVDDataPreprocessor,
    # "create_racmo_masks": CreateRacmoMasks,
    # "extract_freya_glacier_variables_from_racmo": ExtractFreyaGlacierVariablesFromRacmo,
    "plot_singular_vectors": SingularVectors,
    "plot_heterogeneous_correlation": HeterogeneousCorrelation,
    "calculate_xeofs": Xeofs,
    "calculate_complex_rotated_mca": ComplexRotatedMCA,
}
# Deprecated classes
#     "plotmca": PlotMCA,
#     "calc_mca": CalculateMCA,

def functions_completer(prefix, parsed_args, **kwargs):
    obj_class_key = getattr(parsed_args, "object_class", None)
    if obj_class_key not in class_map:
        return []
    cls = class_map[obj_class_key]

    if ":" in prefix:
        func_name, partial_args = prefix.split(":", 1)
        params = get_method_parameters(cls, func_name)
        suggestions = []
        for p in params:
            if p.startswith(partial_args):
                # Show param suggestion with =
                suggestions.append(f"{func_name}:{p}=")
        return suggestions

    methods = get_class_methods(cls)
    return [
        f"{m}:::{get_method_description(cls, m)}".replace(" ", "_")
        for m in methods if m.startswith(prefix)
    ]

def parse_value(val):
    # Try to parse as a Python literal (list, int, float, etc.)
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        pass
    # For bool
    if isinstance(val, str):
        if val.lower() == 'true':
            return True
        if val.lower() == 'false':
            return False
    return val  # fallback as string

def main():
    parser = argparse.ArgumentParser(description="Run methods from processing classes.")

    parser.add_argument(
        "object_class",
        choices=class_map.keys(),
        help="Class to instantiate."
    )

    parser.add_argument(
        "functions",
        nargs="+",
        help="Functions to run, format: func_name[:arg1,arg2,...]"
    ).completer = functions_completer

    # Enable autocomplete BEFORE parsing
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    # Instantiate the chosen class
    obj = class_map[args.object_class]()

    # Run the specified functions
    import re
    def smart_split(arg_str):
        # Split on commas not inside brackets or quotes
        result = []
        current = ''
        depth = 0
        in_single = False
        in_double = False
        for c in arg_str:
            if c == "'" and not in_double:
                in_single = not in_single
            elif c == '"' and not in_single:
                in_double = not in_double
            elif c == '[' and not in_single and not in_double:
                depth += 1
            elif c == ']' and not in_single and not in_double:
                depth -= 1
            if c == ',' and not in_single and not in_double and depth == 0:
                result.append(current.strip())
                current = ''
            else:
                current += c
        if current.strip():
            result.append(current.strip())
        return result

    for func_call in args.functions:
        if ":" in func_call:
            func_name, arg_str = func_call.split(":", 1)
            args_list = smart_split(arg_str)
        else:
            func_name, args_list = func_call, []

        func = getattr(obj, func_name, None)
        if not callable(func):
            print(f"❌ Function '{func_name}' not found in {args.object_class}")
            sys.exit(1)

        # Separate positional and keyword args
        positional_args = []
        kwargs = {}

        for arg in args_list:
            if "=" in arg:
                key, val = arg.split("=", 1)
                val = val.strip("'\"")  # Remove quotes if present
                kwargs[key] = parse_value(val)
            else:
                positional_args.append(arg)

        print(f"====>>>> Running {func_name} with keyword args {kwargs}...")
        func(*positional_args, **kwargs)

if __name__ == "__main__":
    main()
