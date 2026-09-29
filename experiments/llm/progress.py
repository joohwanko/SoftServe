"""Lightweight progress logging and the experiment's QME finiteness check."""

from collections import deque
import json
import time

import torch


class Progress:
    def __init__(self, directory):
        self.directory = directory
        self.losses = deque(maxlen=50)
        self.started = time.perf_counter()
        self.nonfinite_qme = 0

    def initialize(self, model, controller, config, **unused):
        self.controller, self.config = controller, config

    def before_parameter_step(self, step, **unused):
        qn = self.controller.qn
        if qn is not None and step and step % self.config.refresh_interval == 0:
            values = qn.last_qme_residual_A + qn.last_qme_residual_G
            bad = sum(not bool(torch.isfinite(v)) for v in values)
            self.nonfinite_qme += bad
            if bad:
                raise FloatingPointError(f"nonfinite QME residuals: {bad}")

    def after_parameter_step(self, step, gradient, **unused):
        self.losses.append(gradient.loss)
        if (step + 1) % 100 == 0:
            value = dict(
                step=step + 1,
                training_nll_mean50=sum(self.losses) / len(self.losses),
                elapsed_seconds=time.perf_counter() - self.started,
            )
            with (self.directory / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)

    def monitor(self, step, **unused):
        return {"training_nll_mean50": sum(self.losses) / len(self.losses) if self.losses else None}

    def result_fields(self):
        return {"nonfinite_qme_count": self.nonfinite_qme}
