import torch.nn as nn


def bn_filter(module_name, module, param_name, param):
    return isinstance(
        module,
        (
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LazyInstanceNorm1d,
            nn.LazyInstanceNorm2d,
            nn.LazyInstanceNorm3d,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LazyBatchNorm1d,
            nn.LazyBatchNorm2d,
            nn.LazyBatchNorm3d,
        ),
    )


def bias_filter(module_name, module, param_name, param):
    return param_name == "bias"


def bn_or_bias_filter(module_name, module, param_name, param):
    return bn_filter(module_name, module, param_name, param) or bias_filter(
        module_name, module, param_name, param
    )


def pass_all_filter(module_name, module, param_name, param):
    return True


def split_params(model: nn.Module, filters, prefix=""):
    results = []
    for i in range(len(filters)):
        results.append([])
    for module_name, module in model.named_children():
        full_module_name = prefix + module_name
        for param_name, param in module.named_parameters(recurse=False):
            for i, f in enumerate(filters):
                if f(full_module_name, module, param_name, param):
                    results[i].append(param)
                    break
        module_results = split_params(module, filters, full_module_name + ".")
        for i in range(len(filters)):
            results[i] += module_results[i]
    return results


def add_weight_decay(models, weight_decay=1e-5):
    if not isinstance(models, list):
        models = [models]
    params = [[], []]
    for m in models:
        model_params = split_params(m, [bn_or_bias_filter, pass_all_filter])
        params[0] += model_params[0]
        params[1] += model_params[1]
    return [
        {"params": params[0], "weight_decay": 0.0},
        {"params": params[1], "weight_decay": weight_decay},
    ]
