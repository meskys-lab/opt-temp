from fastai.torch_core import flatten_check
from torch import ByteTensor, FloatTensor, Tensor


def misclassified(pred: Tensor, targ: Tensor) -> Tensor:
    "Percentage of misclassified enzymes - thermophilic vs mozophilic"
    _pred, _targ = flatten_check(pred, targ)
    mis_classified = (((_pred >= 55).type(ByteTensor) & (_targ <= 40).type(ByteTensor)) ^
                      ((_pred <= 40).type(ByteTensor) & (_targ >= 55).type(ByteTensor)))
    return mis_classified.type(FloatTensor).mean() * 100


def min_value(pred: Tensor, targ: Tensor) -> Tensor:
    return pred.min()


def max_value(pred: Tensor, targ: Tensor) -> Tensor:
    return pred.max()
