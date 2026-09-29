"""Fail-closed comparison guard for the audited PINN problem mismatch.

The new runner constructed PINNConfig(pde=...) with its default beta=5.
Historical PINNRunConfig.problem() sets Convection beta=40. Matching AD
implementations at beta=5 did not establish parity with that experiment.
Legacy Convection runs remain excluded. Corrected runs must record the complete
resolved problem and match the historical factory before becoming eligible.
"""


def comparison_issue(row):
    cfg = row["config"]
    if cfg.get("task") == "pinn" and "problem" in cfg:
        from .pinn_problem import resolved_problem

        if cfg["problem"] != resolved_problem(cfg):
            return "Recorded PDE configuration differs from the historical problem factory"
        if "problem" in row and row["problem"] != cfg["problem"]:
            return "Executed PDE configuration differs from the immutable configuration"
        return None
    if cfg.get("task") == "pinn" and cfg.get("pde") == "convection":
        return "PDE mismatch: new Convection beta=5; historical comparison beta=40"
    return None


def comparison_eligible(row):
    return comparison_issue(row) is None
