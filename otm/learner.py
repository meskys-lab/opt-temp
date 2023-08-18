from pathlib import Path
from typing import Union

from fastai.data.core import DataLoaders
from fastai.text.all import *
from fastai.metrics import mae, SpearmanCorrCoef
from torch import nn

from otm.model.metrics import misclassified, min_value, max_value


def get_learner(model: nn.Module, dataloaders: DataLoaders, model_dir: Union[str, Path] = 'models') -> Learner:
    learner = Learner(dls=dataloaders,
                      model=model,
                      loss_func=nn.MSELoss(),
                      model_dir=model_dir,
                      metrics=[mae, SpearmanCorrCoef(), misclassified, min_value, max_value])
    return learner
