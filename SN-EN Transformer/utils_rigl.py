# utils_rigl.py
import torch
import torchvision

# Transformer で sparsify したくない層をここで明示的に除外
EXCLUDED_TYPES = (
    torch.nn.BatchNorm2d,
    torch.nn.Embedding,
    torch.nn.LayerNorm,
)

def get_weighted_layers(model, i=0, layers=None, linear_layers_mask=None):
    """
    モデルから「重みを持つ層」のリストを抽出する。
    ResNet 用の実装をベースに、Transformer でも動くように一般化。
    linear_layers_mask: その層が Linear かどうか (1: Linear, 0: その他)
    """
    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []

    items = model._modules.items()
    if i == 0:
        # top-level では model 自身からたどる
        items = [(None, model)]

    for layer_name, p in items:
        if isinstance(p, torch.nn.Linear):
            layers.append([p])
            linear_layers_mask.append(1)
        elif hasattr(p, "weight") and type(p) not in EXCLUDED_TYPES:
            # Conv1d など（Embedding/LayerNorm は EXCLUDED_TYPES で除外）
            layers.append([p])
            linear_layers_mask.append(0)
        elif isinstance(p, torchvision.models.resnet.Bottleneck) or \
             isinstance(p, torchvision.models.resnet.BasicBlock):
            # ResNet ブロック用の再帰
            _, linear_layers_mask, i = get_weighted_layers(
                p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask
            )
        else:
            # その他の nn.Module も再帰的に探索
            _, linear_layers_mask, i = get_weighted_layers(
                p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask
            )

    return layers, linear_layers_mask, i

def get_W(model, return_linear_layers_mask=False):
    """
    モデルから、疎化対象とする weight テンソルのリスト W を返す。
    """
    layers, linear_layers_mask, _ = get_weighted_layers(model)

    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], "weight") else 1
        W.append(layer[idx].weight)

    assert len(W) == len(linear_layers_mask)

    if return_linear_layers_mask:
        return W, linear_layers_mask
    return W

