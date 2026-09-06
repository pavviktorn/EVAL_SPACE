#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Measure each es_val image's content similarity to the TRAINSET, so a content-clean val split can
be carved.

WHY. es_val was carved from the same pool as es_train, while es_test comes from a different source.
Measured consequence: Spearman(val, test) over 511 mean fusions is +0.367 overall but -0.811 within
the top 12 by val -- val screens coarsely and INVERTS for fine selection. Per member, every model
that fine-tunes a backbone loses ground from val to test (A2 -0.0060, gsd -0.0078) while PE-SPC,
which has 15k trainable parameters over a frozen encoder, GAINS +0.0062. That is the signature of val
rewarding trainset-fitting.

If the cause is content shared between train and val, then dropping val images that are near-copies
of trainset images should make val rank more like test. This script computes the evidence; it does
not assume the conclusion -- if val_clean does not improve the rank correlation, the hypothesis is
wrong and gets recorded as such.
"""
import os
import numpy as np
import torch

ES = "/datasets/work/vLLM/temp/EVAL_SPACE"


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    zt = np.load(f"{ES}/cache/es_train_0.npz", allow_pickle=True)
    zv = np.load(f"{ES}/cache/es_val_0.npz", allow_pickle=True)
    Ft = torch.from_numpy(zt["feats"]).to(dev).half()
    Fv = torch.from_numpy(zv["feats"]).to(dev).half()
    Ft = torch.nn.functional.normalize(Ft.float(), dim=1).half()
    Fv = torch.nn.functional.normalize(Fv.float(), dim=1).half()
    print(f"[valclean] train {tuple(Ft.shape)}  val {tuple(Fv.shape)} on {dev}", flush=True)

    maxcos = torch.zeros(Fv.shape[0], device=dev, dtype=torch.float32)
    VB, TB = 4096, 200_000
    for i in range(0, Fv.shape[0], VB):
        vb = Fv[i:i + VB]
        best = torch.full((vb.shape[0],), -2.0, device=dev, dtype=torch.float32)
        for j in range(0, Ft.shape[0], TB):
            sim = (vb @ Ft[j:j + TB].t()).float()
            best = torch.maximum(best, sim.max(dim=1).values)
        maxcos[i:i + VB] = best
        if (i // VB) % 4 == 0:
            print(f"[valclean] {i + vb.shape[0]:,}/{Fv.shape[0]:,}", flush=True)
    mc = maxcos.cpu().numpy()
    np.save(f"{ES}/work/es_val_maxcos_to_train.npy", mc)
    paths = zv["paths"]
    np.save(f"{ES}/work/es_val_paths.npy", paths)
    print(f"\n[valclean] max-cosine of each val image to its nearest TRAIN image:")
    for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        print(f"    p{int(q*100):>2} = {np.quantile(mc, q):.4f}")
    for thr in (0.90, 0.95, 0.98, 0.99, 0.995):
        n = int((mc >= thr).sum())
        print(f"    >= {thr:.3f}: {n:>7,} of {len(mc):,} ({100*n/len(mc):5.2f}%) would be dropped")
    print(f"[valclean] -> work/es_val_maxcos_to_train.npy")


if __name__ == "__main__":
    main()
