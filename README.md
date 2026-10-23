# PMT RGMT Transformer 20k

[Author project](https://github.com/Mondo-Robotics/PMT) · [Checkpoint](https://huggingface.co/datasets/aCodeDog/PMT-assets/blob/38f9f7dba893385c6219620b72314396247c9e82/checkpoints/pretrained/rgmt_flat.pt) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-pmt-rgmt.git
cd motion-challenge-pmt-rgmt
uv sync
uv run evaluate.py
```

One full rollout per motion produces success, completion and all six MPBPE values
in one verified `result.json`. MuJoCo 3.17 CPU batch; `--jobs 8` sets rollout and
verification parallelism. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This repository adapts model inference. The pinned root `motion-challenge`
submodule owns datasets, physics, metrics and verification. See `--help` for options.
