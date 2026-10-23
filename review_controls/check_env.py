#!/usr/bin/env python3
"""Check the Python/GPU environment, then download MNIST, EMNIST (balanced) and
the LPIPS/VGG weights once into the shared directories, so that the three
shards never download concurrently over NFS.

    XORLAB_DATA=/nfs/xor/data TORCH_HOME=/nfs/xor/torch_home python check_env.py
"""
import os
import platform
import sys
import time

ok = True


def say(tag, msg):
    print(f"[{tag:4s}] {msg}", flush=True)


try:
    import torch
    say("ok", f"torch {torch.__version__}  python {sys.version.split()[0]}  {platform.machine()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        say("ok", f"CUDA {torch.version.cuda}: {p.name}  sm_{p.major}{p.minor}  "
                  f"{p.total_memory / 2**30:.0f} GiB")
        a = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize(); t = time.time()
        for _ in range(10):
            a @ a
        torch.cuda.synchronize()
        say("ok", f"fp32 matmul {10 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOP/s (TF32 off)")
        x = torch.randn(256, 1, 28, 28, device="cuda")
        torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(
            torch.nn.functional.conv2d(x, torch.randn(28, 1, 3, 3, device="cuda"), padding=1),
            4, return_indices=True), 4)
        say("ok", "conv2d / max_pool2d / max_unpool2d on GPU")
    else:
        say("WARN", "torch.cuda.is_available() is False: runs will use CPU (slow, esp. CNN)")
except Exception as e:
    say("FAIL", f"torch: {e}"); ok = False

for mod in ("torchvision", "numpy", "matplotlib", "lpips"):
    try:
        m = __import__(mod)
        say("ok", f"{mod} {getattr(m, '__version__', '')}")
    except Exception as e:
        say("FAIL" if mod != "lpips" else "WARN", f"{mod}: {e}"
            + ("  (pip install lpips; or run grid.py --no-lpips)" if mod == "lpips" else ""))
        ok = ok and mod == "lpips"

root = os.environ.get("XORLAB_DATA", "./data")
try:
    from torchvision import datasets
    for tr in (True, False):
        n = len(datasets.MNIST(root, train=tr, download=True))
        say("ok", f"MNIST {'train' if tr else 'test'}: {n}")
        n = len(datasets.EMNIST(root, split="balanced", train=tr, download=True))
        say("ok", f"EMNIST-balanced {'train' if tr else 'test'}: {n}")
except Exception as e:
    say("FAIL", f"dataset download into {root}: {e}\n"
               "       If the NIST EMNIST URL is down, fetch gzip.zip manually into "
               f"{root}/EMNIST/raw/ and re-run.")
    ok = False

try:
    import lpips
    lpips.LPIPS(net="vgg", verbose=False)
    say("ok", f"LPIPS/VGG weights cached (TORCH_HOME={os.environ.get('TORCH_HOME', '~/.cache/torch')})")
except Exception as e:
    say("WARN", f"LPIPS init failed: {e}")

print("\nREADY" if ok else "\nNOT READY — fix the FAIL lines above")
sys.exit(0 if ok else 1)
