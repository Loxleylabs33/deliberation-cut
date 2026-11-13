#!/usr/bin/env python3
"""Lay out the files of this release the way the scripts in this folder expect them.

    python3 setup_workdir.py WORKDIR [--force]
    export CUT_ROOT=WORKDIR

Copies the small files (routers, question vectors, labels, per-item results) and links the 2.2 GB piece, so that the
scripts can read and overwrite their own copies without touching the release. Existing files are left alone unless you
pass --force. Datasets and models are not part of the release; see README.md for fetch_data.sh and for where the
Qwen model folders go.

  router/embed.pt                                 -> hyper/embed.pt
  router/embed_long_budget.pt                     -> hyper/c32/x.pt
  router/router_pool440_h3b/router_v2_pool440.pt  -> hyper/router_v2.pt
  router/router_pool32k_h5/router_v2_pool32k.pt   -> hyper/h5/router_v2.pt
  router/pool_labels/pool440/*.jsonl              -> hyper/pool/
  router/pool_labels/pool32k/*.jsonl              -> hyper/pool_text32/
  results/items/<label>/*                         -> step4/<label>/
  piece/P_q80.safetensors                         -> hyper/P_q80.safetensors  (symlink)
  code/DATA_SHA256.txt                            -> DATA_SHA256.txt
"""
import glob
import os
import shutil
import sys

RELEASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def put(src, dst, force, link=False):
    if os.path.lexists(dst) and not force:
        print("exists, left alone:", dst)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    if link:
        os.symlink(src, dst)
    else:
        shutil.copy(src, dst)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    if len(args) != 1:
        sys.exit(__doc__)
    work = os.path.abspath(args[0])
    r = lambda *p: os.path.join(RELEASE, *p)  # noqa: E731
    w = lambda *p: os.path.join(work, *p)  # noqa: E731
    for d in ("data", "models", "logs"):
        os.makedirs(w(d), exist_ok=True)
    put(r("router", "embed.pt"), w("hyper", "embed.pt"), force)
    put(r("router", "embed_long_budget.pt"), w("hyper", "c32", "x.pt"), force)
    put(r("router", "router_pool440_h3b", "router_v2_pool440.pt"), w("hyper", "router_v2.pt"), force)
    put(r("router", "router_pool32k_h5", "router_v2_pool32k.pt"), w("hyper", "h5", "router_v2.pt"), force)
    for f in sorted(glob.glob(r("router", "pool_labels", "pool440", "*.jsonl"))):
        put(f, w("hyper", "pool", os.path.basename(f)), force)
    for f in sorted(glob.glob(r("router", "pool_labels", "pool32k", "*.jsonl"))):
        put(f, w("hyper", "pool_text32", os.path.basename(f)), force)
    for d in sorted(glob.glob(r("results", "items", "*"))):
        for f in sorted(os.listdir(d)):
            put(os.path.join(d, f), w("step4", os.path.basename(d), f), force)
    put(r("piece", "P_q80.safetensors"), w("hyper", "P_q80.safetensors"), force, link=True)
    put(r("code", "DATA_SHA256.txt"), w("DATA_SHA256.txt"), force)
    print(f"ready: export CUT_ROOT={work}")


if __name__ == "__main__":
    main()
