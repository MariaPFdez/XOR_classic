# Option B — NGC PyTorch container (supported path on DGX Spark).
# To reproduce the paper runs exactly, set NGC_TAG to a tag whose PyTorch
# matches the version recorded in results/*/seed*.json (field "torch",
# summarised in results/summary.md, section "Environment").
ARG NGC_TAG=25.09-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}
RUN pip install --no-cache-dir lpips
