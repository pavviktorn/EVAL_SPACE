#!/usr/bin/env python
"""Build the clean testset: images in no_delete_mids_train that the trainset never used, with content
duplicates removed.

Path-difference alone is NOT enough. Measured on this data earlier: the SAME image is stored under
different paths (one cluster is 231 byte-identical copies of one real image across separate user
directories), so a "new path" can still be an old image. Two removals are applied:

  * EXACT   -- feature-hash identity against the trainset, and within the candidate pool itself.
               Hashing, never a cosine cut: the fp16 cache stores vectors with norms 0.9956-1.0049,
               so a raw dot of two identical rows is ||v||^2 in [0.9913, 1.0098] and a >=0.9999 cut
               misses ~40% of them.
  * NEAR    -- true cosine (fp32-renormalised) above a threshold chosen from the MEASURED
               distribution, with the full sensitivity table printed so the choice is auditable.

Candidates are compared against ALL original trainset content (940,299 distinct vectors), not just the
940,014 kept after de-duplication. That is deliberate and conservative: a dropped duplicate still has
an identical twin inside the clean trainset, and the contradictory-label clusters that were removed
entirely are content no testset should contain either.
"""
import argparse, collections, hashlib, json, os, sys
import numpy as np
import torch

PS = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(PS, "spc"))
import cache_io                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cand", default="cache/cand")
ap.add_argument("--thresh", type=float, default=None,
                help="near-duplicate cosine cut; default: chosen from the measured distribution")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default="work/testset_clean_images.json")
ap.add_argument("--report", default="runs/testset_dedup.json")
a = ap.parse_args()

tr = cache_io.load_shards(os.path.join(PS, "cache/train"))
cd = cache_io.load_shards(a.cand)
print(f"[clean] trainset rows {len(tr['labels']):,} | candidates {len(cd['labels']):,}")
bad = int((cd["ok"] == 0).sum())
if bad:
    raise SystemExit(f"[clean] {bad} candidate features came from unreadable images -- fix the data")

htr = set(hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in tr["feats"])
hcd = [hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in cd["feats"]]
exact_tr = np.array([h in htr for h in hcd])
seen, dup_self = set(), np.zeros(len(hcd), bool)
for i, h in enumerate(hcd):
    if h in seen:
        dup_self[i] = True
    else:
        seen.add(h)
print(f"[clean] exact duplicates of TRAINSET content : {int(exact_tr.sum()):,} "
      f"({exact_tr.mean()*100:.3f}%)")
print(f"[clean] exact duplicates WITHIN the candidates: {int(dup_self.sum()):,} "
      f"({dup_self.mean()*100:.3f}%)")


def l2(x):
    x = x.to(torch.float32)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


D = a.device
ftr = l2(torch.from_numpy(tr["feats"]).to(D)).to(torch.float16)
fcd = l2(torch.from_numpy(cd["feats"]).to(D)).to(torch.float16)
best = np.empty(len(fcd), np.float32)
for i in range(0, len(fcd), 4096):
    d = fcd[i:i + 4096]
    m = torch.full((d.shape[0],), -1.0, device=D, dtype=torch.float16)
    for j in range(0, len(ftr), 262144):
        m = torch.maximum(m, (d @ ftr[j:j + 262144].t()).max(1).values)
    best[i:i + 4096] = m.float().cpu().numpy()
del ftr, fcd
torch.cuda.empty_cache()

pct = [1, 5, 25, 50, 75, 90, 95, 99]
print(f"[clean] candidate max-cosine to TRAINSET: "
      + " ".join(f"p{p}={np.percentile(best,p):.4f}" for p in pct))
lab = cd["labels"].astype(np.int64)
CN = {0: "real", 1: "pad", 2: "deepfake"}
table = {}
print(f"[clean] {'thresh':>8} {'removed':>10} {'%':>7} {'kept':>10} "
      + " ".join(f"{CN[c]:>9}" for c in (0, 1, 2)))
for t in (0.9999, 0.999, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90):
    rm = exact_tr | dup_self | (best >= t)
    kp = ~rm
    table[str(t)] = {"removed": int(rm.sum()), "kept": int(kp.sum()),
                     "kept_by_class": {CN[c]: int((kp & (lab == c)).sum()) for c in (0, 1, 2)}}
    print(f"[clean] {t:>8} {int(rm.sum()):>10,} {rm.mean()*100:>6.2f}% {int(kp.sum()):>10,} "
          + " ".join(f"{int((kp & (lab==c)).sum()):>9,}" for c in (0, 1, 2)))

# Choose the threshold from the distribution rather than by habit: take the tightest cut that still
# leaves a testset with a usable number of every class. 0.99 is the default preference because above
# it an image is almost certainly the same picture re-encoded, whereas the 0.95 band is dominated by
# genuinely distinct frames from the same capture rig.
thresh = a.thresh
if thresh is None:
    for t in (0.99, 0.98, 0.97, 0.96, 0.95):
        k = table[str(t)]["kept_by_class"]
        if min(k.values()) >= 5000:
            thresh = t
            break
    thresh = thresh or 0.95
rm = exact_tr | dup_self | (best >= thresh)
keep = np.flatnonzero(~rm)
kh = collections.Counter(int(lab[i]) for i in keep)
print(f"\n[clean] CHOSEN near-duplicate threshold {thresh}")
print(f"[clean] CLEAN TESTSET: {len(keep):,} images "
      f"(real {kh[0]:,} / pad {kh[1]:,} / deepfake {kh[2]:,})")
print(f"[clean]   removed: {int(exact_tr.sum()):,} exact-vs-train, {int(dup_self.sum()):,} self-dup, "
      f"{int(((best >= thresh) & ~exact_tr & ~dup_self).sum()):,} near-dup only")

recs = [{"image": str(cd["paths"][i]), "label": int(lab[i])} for i in keep]
json.dump(recs, open(a.out, "w"))
json.dump({"n_candidates": int(len(lab)), "chosen_threshold": float(thresh),
           "exact_vs_train": int(exact_tr.sum()), "self_dup": int(dup_self.sum()),
           "kept": int(len(keep)),
           "kept_by_class": {CN[c]: int(kh[c]) for c in (0, 1, 2)},
           "max_cos_percentiles": {f"p{p}": float(np.percentile(best, p)) for p in pct},
           "sensitivity": table}, open(a.report, "w"), indent=1)
np.save("work/testset_cand_maxcos.npy", best)
print(f"[clean] -> {a.out}\n[clean] -> {a.report}")
