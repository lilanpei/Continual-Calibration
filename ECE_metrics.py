from avalanche.evaluation import Metric
from avalanche.evaluation import GenericPluginMetric
import torch
import numpy as np


class ECE(Metric[float]):

    def __init__(self, bins=None):
        super().__init__()

        if bins is not None:
            assert all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)), "Bins must be sorted in ascending order"
            assert max(bins) <= 1.0 and min(bins) >= 0.0, "Bins must be in [0, 1]"

        if bins is None:
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.bins = bins
        self.example_confidences = None
        self.example_accuracy = None

    @torch.no_grad()
    def update(self, predicted_y, true_y):
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)
        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        assert len(predicted_y.shape) > 1, "ECE needs the entire logit vector, not labels."

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        cfd = torch.softmax(predicted_y, dim=-1)
        cfd = cfd.max(dim=-1)[0]
        acc = (predicted_y.max(dim=-1)[1] == true_y).float()

        if self.example_confidences is None:
            self.example_confidences = cfd
        else:
            self.example_confidences = torch.cat([self.example_confidences, cfd], dim=0)
        if self.example_accuracy is None:
            self.example_accuracy = acc
        else:
            self.example_accuracy = torch.cat([self.example_accuracy, acc], dim=0)

    def result(self) -> float:
        ece = {}

        max_bin = self.bins[-1]
        for min_bin in self.bins[:-1]:
            mask = torch.logical_and(self.example_confidences <= max_bin, self.example_confidences > min_bin)
            n_examples = mask.sum().item()
            if n_examples > 0:
                current_conf, current_pred = self.example_confidences[mask], self.example_accuracy[mask]
                avg_conf = current_conf.mean()
                avg_acc = current_pred.mean()
                ece[min_bin] = (avg_acc.cpu().numpy(), avg_conf.cpu().numpy(), n_examples)
            else:
                ece[min_bin] = (0, 0, 0)
            # update bin
            max_bin = min_bin

        return sum([(float(nex) / float(self.example_confidences.shape[0])) * np.abs(acc - conf) for acc, conf, nex in ece.values()])

    def reset(self):
        self.example_confidences = None
        self.example_accuracy = None


class GenericECE(GenericPluginMetric[float, ECE]):
    def __init__(self, reset_at, emit_at, mode, bins=None):
        super().__init__(ECE(bins=bins), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)


class ExperienceECE(GenericECE):
    def __init__(self, bins=None):
        super().__init__(
            reset_at="experience", emit_at="experience", mode="eval",
            bins=bins
        )

    def __str__(self):
        return "ECE_Exp"


