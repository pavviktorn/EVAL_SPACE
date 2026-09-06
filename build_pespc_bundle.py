#!/usr/bin/env python
"""Assemble PE feature bundles for the EVAL_SPACE splits, reusing already-extracted features.

No new extraction is needed: the clean trainset and val split are subsets of the paths in
PAAS_simplicity/cache/train_*.npz (1,360,956 rows), and the clean testset is a subset of
EVAL_SPACE/cache/cand_*.npz (421,000 candidates). This is why PE-SPC retrains on the identical clean
data in about a minute while the CLIP and MLLM detectors need ~17 GPU-hours -- the encoder is frozen,
so its output is cacheable once and reusable for any split of the same images.
"""
import json, os, sys
import numpy as np

PS = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(PS, "spc"))
import cache_io                                        # noqa: E402

OUT = "/datasets/work/vLLM/temp/EVAL_SPACE/cache"


def gather(split_json, sources, name):
    want = {r["image"]: int(r["label"]) for r in json.load(open(split_json))}
    feats, labs, paths = [], [], []
    seen = set()
    for prefix in sources:
        c = cache_io.load_shards(prefix)
        p = c["paths"].tolist()
        # dedupe row-by-row, not per-prefix: the source cache holds up to TEN rows per path (the
        # manifest repeats 41,557 paths 10x), so a per-prefix filter returned MORE features than
        # wanted images. Also note the PE forward is deterministic to cosine 1.0 but not bit-identical
        # for ~0.002% of images (different batch composition changes the fp16 reduction order), so
        # content-hashing alone can leave two "contents" for one path -- path-uniqueness is the
        # invariant that matters when the unit of training is an image.
        idx = []
        for k, pp in enumerate(p):
            if pp in want and pp not in seen:
                seen.add(pp); idx.append(k)
        if idx:
            feats.append(c["feats"][idx])
            labs.append(np.array([want[p[k]] for k in idx], np.int8))
            paths.append(np.array([p[k] for k in idx], dtype=object))
        del c
    if not feats:
        raise SystemExit(f"[bundle] {name}: no features found for any of its {len(want):,} images")
    F = np.concatenate(feats); L = np.concatenate(labs); P = np.concatenate(paths)
    miss = len(want) - len(F)
    print(f"[bundle] {name}: {len(F):,}/{len(want):,} images have cached features "
          f"({miss:,} missing)")
    if miss:
        raise SystemExit(f"[bundle] {name}: {miss:,} images have NO cached feature. Extract them "
                         f"before training, or the split silently shrinks.")
    out = os.path.join(OUT, f"{name}_0.npz")
    np.savez(out, feats=F, labels=L, paths=P, ok=np.ones(len(L), np.int8),
             index=np.arange(len(L)), unreadable=0)
    print(f"[bundle] -> {out}  {F.shape}")
    return len(F)


TR = [os.path.join(PS, "cache/train")]
CD = ["/datasets/work/vLLM/temp/EVAL_SPACE/cache/cand"]
gather("work/trainset_final_images.json", TR, "es_train")
gather("work/valset_images.json", TR, "es_val")
gather("work/testset_clean_images.json", CD, "es_test")
