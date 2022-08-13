import torch
from model import BaseAQFModel


class EnsembleModule:
    def __init__(self, *models: BaseAQFModel, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.models = [model.to(self.device) for model in models]

    def predict(self, dt):
        avg_out = None

        for model in self.models:
            out = model.predict(dt)

            if avg_out is None:
                avg_out = out
            else:
                avg_out = avg_out + out

        avg_out = avg_out / len(self.models)

        return avg_out