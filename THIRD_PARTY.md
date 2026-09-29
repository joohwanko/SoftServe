# Third-party code and experiment sources

Third-party authorship notices and licenses are retained with the corresponding code. No datasets or pretrained weights are redistributed here.

- **PIDM:** `experiments/pidm/upstream/` contains the required modules from [PhysicsInformedDiffusionModels](https://github.com/jhbastek/PhysicsInformedDiffusionModels), commit `733aeb0fe8e56f8d1b3aaf9ff24f340438c20d9a`. The MIT license is retained in that directory. Package imports were adjusted; a missing functional import in an unused data helper was supplied. The mechanics configuration is from the same release. Training equations and the U-Net are unchanged. The training runner omits final checkpoint generation and generated-design visualizations.
- **Official SOAP:** `experiments/llm/soap/` contains the [SOAP implementation](https://github.com/nikhilvyas/SOAP) and its MIT license. This is the SOAP used by the language-model runner.
- **Meta SOAP:** the other SOAP experiments use [facebookresearch/optimizers](https://github.com/facebookresearch/optimizers), pinned in `pyproject.toml`. Their configured averaging, parameter routing, and preconditioning frequencies are preserved rather than replaced by the language-model implementation.
- **PyTorch Muon:** small-task and GPT baselines use `torch.optim.Muon`. PIDM groups the same NS5 arithmetic by matrix shape, with Adam on the remaining parameters.
- **Small PINNs:** equations and benchmark conventions follow [Challenges in Training PINNs: A Loss Landscape Perspective](https://github.com/pratikrathore8/opt_for_pinns). The task definitions use Convection β=40, Reaction ρ=5, and Wave β=5.
- **MNIST autoencoder / K-FAC / K-BFGS(L):** the architecture and structured-baseline recipes follow Goldfarb, Ren, and Bahamou, *Practical Quasi-Newton Methods for Training Deep Neural Networks*. RNN and PINN K-BFGS(L) use the documented shared-weight mean-pair extension; they are not claimed to be an official recurrent K-BFGS implementation.
- **FineWeb:** [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), `sample-10BT`, tokenized with GPT-2. Dataset terms apply independently of this code.

## PirateNet exclusion

The upstream [jaxpi/PirateNet license](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/LICENSE) requires prior written approval for third-party distribution of the software or modifications. Consequently, the PirateNet implementation/port is not included in this repository pending permission. `configs/piratenet.json` records the selected numerical settings only; it is not a runnable reproduction by itself. Other experiment runners do not depend on PirateNet.
