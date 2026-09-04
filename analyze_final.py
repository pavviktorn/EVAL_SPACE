#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Build runs/final_comparison.json: all five detectors, one trainset, one testset, one harness.

Every model here was retrained on EVAL_SPACE/dataset/trainset.json and is scored on
EVAL_SPACE/dataset/testset.json, which shares no image -- and (per check_old_eval_leak.py) no image
CONTENT -- with the trainset or with the old selection set.

Two families of number are reported and they must not be mixed up:

  THRESHOLD-FREE (bin_auc, ap, eer)   ranking quality; no operating point involved.
  VAL-FITTED, TEST-APPLIED            tau chosen on es_val at a real-recall target, then applied
                                      unchanged to es_test. This is the only deployable accuracy.
                                      Anything fitted on the test set is labelled as such and is NOT
                                      comparable to it.

Members are aligned BY IMAGE PATH, not by row order: the shards are strided and FFAA can drop an
unreadable image that another member kept, so positional alignment would silently compare different
images. Only images every member scored are used, and the dropped count is reported.
"""
import argparse, glob, json, os, sys
import numpy as np

PS = "/datasets/work/vLLM/temp/PAAS_simplicity"
ES = "/datasets/work/vLLM/temp/EVAL_SPACE"
sys.path.insert(0, os.path.join(PS, "spc"))
import metrics as M                                                 # noqa: E402

NICE = {"ensemble_fake": "9c ensemble (mean of the members the deploy loaded)",
        "a1_9c_fake": "A1_9c (Effort SVD only)",
        "a2_9c_fake": "A2_9c (Effort SVD + GenD)",
        "a3_9c_fake": "A3_9c (full MIDS++: SVD + GenD + ForAda)",
        "gsd_fake": "GSD", "selop_fake": "SeLop",
        "ffaa_fake": "FFAA (Qwen3.5-4B + MIDS-4c)",
        "pespc_default": "PE-SPC (Exp-24 recipe, FROZEN - the comparison row)",
        "pespc_paper": "PE-SPC (paper recipe: H0, lr 2e-5)",
        "pespc_valsel_h3": "PE-SPC* (H3 K=4, lr 1e-2 - lr chosen on es_val)",
        "pespc_valsel_h4": "PE-SPC* (H4 MLP, lr 1e-2 - lr chosen on es_val)"}
# The starred rows had their learning rate selected on es_val. The baselines did not get that, so the
# star is load-bearing: it marks the rows that are NOT like-for-like. See runs/recipe_asymmetry.json.


E9_LOADED = {}          # filled by load_split from the score shards
PER_MEMBER_COL = {"A1_9c": "a1_9c_fake", "A2_9c": "a2_9c_fake", "A3_9c": "a3_9c_fake"}
PER_MEMBER_OF = {v: k for k, v in PER_MEMBER_COL.items()}


def load_split(pattern):
    """Merge strided shards into {path: {member: score}} + {path: label}.

    `pattern` may be one glob or a list of globs. Shards carrying DIFFERENT member sets are merged as
    extra COLUMNS -- score_9c_members.py scores individual 9-class members separately (because
    paas/pipeline.py only builds the ones named in fusion.components) and score_gsd_faithful.py does
    the same for the paper-faithful arm. Shards sharing a member set must still agree on it exactly,
    since that is the signature of two incompatible scoring runs in one directory.
    """
    pats = [pattern] if isinstance(pattern, str) else list(pattern)
    files = sorted({f for pat in pats for f in glob.glob(pat)})
    if not files:
        raise SystemExit(f"[final] no score shards match {pats}")
    per, lab, unread = {}, {}, 0
    gin = {}   # n_in PER shard group: summing across groups multiplies the split size
    groups, owner = {}, {}
    for f in files:
        d = json.load(open(f))
        mem = tuple(d["members"])
        groups.setdefault(mem, []).append(os.path.basename(f))
        for k in mem:
            if k in owner and owner[k] != mem:
                raise SystemExit(f"[final] member {k} appears in two different shard groups "
                                 f"({owner[k]} and {mem}) -- refusing to guess which is current")
            owner[k] = mem
        for _n in (d.get("ensemble9_members_loaded") or []):
            E9_LOADED[_n] = True
        # score_final.py writes `unreadable` as a LIST of {image, err}; score_9c_members.py and
        # score_gsd_faithful.py write it as a COUNT. Accept both rather than crashing on the
        # merge of shard groups this loader exists to support.
        _u = d.get("unreadable", 0)
        unread += len(_u) if isinstance(_u, (list, tuple)) else int(_u or 0)
        gin[mem] = gin.get(mem, 0) + int(d.get("n_in", 0))
        s = d["scores"]
        for i, p in enumerate(s["image"]):
            l = int(s["label"][i])
            if lab.get(p, l) != l:
                raise SystemExit(f"[final] {p} is labelled {lab[p]} in one shard and {l} in another")
            lab[p] = l
            per.setdefault(p, {}).update({k: float(s[k][i]) for k in mem})
    members = list(dict.fromkeys(k for mem in groups for k in mem))
    nin = max(gin.values()) if gin else 0
    print(f"[final] {len(files)} shard(s) in {len(groups)} group(s): {len(per):,} images "
          f"(largest group covered {nin:,}, unreadable {unread}) members={members}")
    return per, lab, members, unread


def vectors(per, lab, members):
    """Only images that every member scored, with no NaN. Report what that costs."""
    paths = sorted(p for p, d in per.items()
                   if all(k in d and not np.isnan(d[k]) for k in members))
    dropped = len(per) - len(paths)
    y = np.array([lab[p] for p in paths], np.int64)
    X = {k: np.array([per[p][k] for p in paths], np.float64) for k in members}
    return paths, y, X, dropped


def fuse(Xv, yv, Xt, keys, l2=1.0, steps=400):
    """Logistic regression over member fake scores, FITTED ON VAL, applied to test.

    Why a fitted fusion and not an average: the members are on different scales and have very
    different reliabilities, and an unweighted average silently lets the weakest member veto the
    strongest. Why logistic regression and not something larger: with 4-6 features and ~50k val rows,
    anything bigger would fit the val split rather than the problem, and the whole point of this
    directory is to stop measuring selection fit.

    Standardisation uses VAL statistics only, and they are applied unchanged to test -- computing test
    statistics would leak the test distribution into the transform, which is the same mistake as
    fitting a threshold on test.

    Returns (val_scores, test_scores, weights_dict). Both score vectors are probabilities of FAKE, so
    they drop straight into the same metric block as any member.
    """
    import torch
    V = torch.tensor(np.stack([Xv[k] for k in keys], 1), dtype=torch.float64)
    T = torch.tensor(np.stack([Xt[k] for k in keys], 1), dtype=torch.float64)
    mu, sd = V.mean(0, keepdim=True), V.std(0, keepdim=True).clamp_min(1e-9)
    V = (V - mu) / sd
    T = (T - mu) / sd                                    # VAL statistics, applied to test
    y = torch.tensor((yv != M.REAL).astype(np.float64))
    w = torch.zeros(len(keys), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = V @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + l2 * (w ** 2).sum() / len(y)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        sv = torch.sigmoid(V @ w + b).numpy()
        st = torch.sigmoid(T @ w + b).numpy()
        wd = {k: float(v) for k, v in zip(keys, w)}
        wd["_bias"] = float(b)
    return sv, st, wd


def old_sel_neardup_paths(thr=0.98):
    """Paths in es_test whose PE-feature cosine to the OLD baseline selection set reaches `thr`.

    The baselines' WEIGHTS are retrained here, but their RECIPES were selected on
    testset/testset_mids/mids_testset.json (30,197 images). check_old_eval_leak.py measured 0 exact
    duplicates of that set in es_test and 2,907 images (1.203%) at cosine >= 0.98, max 0.9997 -- i.e.
    adjacent frames of the same captures, not the same files. PE-SPC's own selection data (DEV-A, carved
    from the trainset pool) is already 0.98-deduplicated against es_test by build_clean_testset.py, so
    that 1.2% is an advantage only the baselines get. Reporting a view with it removed makes the
    comparison symmetric; reporting only that view would hide the headline number, so both are kept.
    """
    mx = os.path.join(ES, "work/es_test_maxcos_to_old_eval.npy")
    bundle = os.path.join(ES, "cache/es_test_0.npz")
    if not (os.path.exists(mx) and os.path.exists(bundle)):
        return None
    m = np.load(mx)
    P = np.load(bundle, allow_pickle=True)["paths"]
    if len(m) != len(P):
        raise SystemExit(f"[final] maxcos has {len(m):,} rows but es_test bundle has {len(P):,}")
    return set(P[m >= thr].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-glob", nargs="+",
                    default=[os.path.join(ES, "runs/scores/val_*.json"),
                             os.path.join(ES, "runs/scores_9c/val_*.json"),
                             os.path.join(ES, "runs/scores_faithful/val_*.json"),
                             os.path.join(ES, "runs/scores_anchorswap/val_*.json")])
    ap.add_argument("--test-glob", nargs="+",
                    default=[os.path.join(ES, "runs/scores/test_*.json"),
                             os.path.join(ES, "runs/scores_9c/test_*.json"),
                             os.path.join(ES, "runs/scores_faithful/test_*.json"),
                             os.path.join(ES, "runs/scores_anchorswap/test_*.json")])
    ap.add_argument("--pespc", default=os.path.join(ES, "runs/pespc/pespc_eval_space.json"))
    ap.add_argument("--out", default=os.path.join(ES, "runs/final_comparison.json"))
    ap.add_argument("--targets", nargs="*", type=float, default=[0.95, 0.98, 0.99])
    a = ap.parse_args()

    vper, vlab, members, vunread = load_split(a.val_glob)
    tper, tlab, tmembers, tunread = load_split(a.test_glob)
    if members != tmembers:
        raise SystemExit(f"[final] val scored {members} but test scored {tmembers}")
    # DROP COLUMNS THAT ARE ENTIRELY ABSENT, LOUDLY. vectors() keeps only images where EVERY member
    # has a finite score, so ONE all-NaN member empties the intersection and every metric below
    # becomes NaN. That happened for real: paas/pipeline.py loads only the 9c members named in
    # fusion.components, so a1_9c_fake and a3_9c_fake were NaN for all 292,051 images and the
    # intersection came out zero. Dropping them here keeps the rest of the table usable; the message
    # says which member is missing so it reads as "absent", never as "performed badly".
    def _all_nan(per, mem):
        vals = [rec.get(mem) for rec in per.values()]
        return not any(v is not None and np.isfinite(v) for v in vals)
    dead = [m for m in members if _all_nan(vper, m) or _all_nan(tper, m)]
    if dead:
        print(f"[final] DROPPING {len(dead)} member(s) with no finite score on at least one split: "
              f"{dead}\n"
              f"        These were NOT scored -- do not read their absence as poor performance. If "
              f"they are 9c members, paas/pipeline.py loads only those in fusion.components; score "
              f"them with score_9c_members.py and merge, or add them to fusion.components.")
        members = [m for m in members if m not in dead]
        if not members:
            raise SystemExit("[final] REFUSING: every member column is empty.")
    vpaths, yv, Xv, vdrop = vectors(vper, vlab, members)
    tpaths, yt, Xt, tdrop = vectors(tper, tlab, members)
    if set(vpaths) & set(tpaths):
        raise SystemExit(f"[final] {len(set(vpaths)&set(tpaths)):,} images are in BOTH val and test")
    print(f"[final] common-image val {len(vpaths):,} (dropped {vdrop}) | "
          f"test {len(tpaths):,} (dropped {tdrop})")
    # A split missing a class produces NaN for auc/eer and for every real-recall-anchored number, and
    # NaN propagates quietly through the whole table -- a synthetic dry run of this script printed
    # auc=nan for all four members and still wrote a complete-looking report. Fail here instead.
    for nm, yy in (("val", yv), ("test", yt)):
        hist = np.bincount(yy, minlength=3)
        if hist[M.REAL] == 0 or (hist[M.PAD] + hist[M.DEEPFAKE]) == 0:
            raise SystemExit(f"[final] REFUSING: the {nm} set has real={hist[M.REAL]:,} "
                             f"pad={hist[M.PAD]:,} deepfake={hist[M.DEEPFAKE]:,}. Every "
                             f"real-recall-anchored metric would be NaN and the table would still "
                             f"render. Check the manifest and the shard globs.")
        print(f"[final] {nm} class mix: real={hist[M.REAL]:,} pad={hist[M.PAD]:,} "
              f"deepfake={hist[M.DEEPFAKE]:,}")

    near = old_sel_neardup_paths()
    views = {"full": None}
    if near:
        views["no_oldsel_neardup"] = near
        n_rm = len([p for p in tpaths if p in near])
        print(f"[final] view no_oldsel_neardup: removing {n_rm:,} of {len(tpaths):,} test images "
              f"(cos>=0.98 to the old baseline selection set)")

    # The shards' ensemble9_members_loaded came from the CONFIG FILE in the version of score_final.py
    # that wrote them, so it can claim members the pipeline never built. Reconcile it against the
    # per-member columns that actually carry data, and against ensemble_fake itself: if the fused
    # column is numerically identical to one member's column, the "ensemble" IS that member.
    _alive = {PER_MEMBER_OF.get(m) for m in members if m in PER_MEMBER_OF}
    _alive.discard(None)
    if E9_LOADED and _alive and set(E9_LOADED) - _alive:
        _phantom = sorted(set(E9_LOADED) - _alive)
        print(f"[final] CORRECTING the ensemble label: the shards claim {sorted(E9_LOADED)} but "
              f"{_phantom} have no scored column, so the fused 'ensemble' is really "
              f"{sorted(_alive)}. The claim came from the config file, not the runtime.")
        for _p in _phantom:
            E9_LOADED.pop(_p, None)
    # BIT-IDENTITY OUTRANKS THE CLAIM. Once A1/A3 were scored separately their columns exist, so the
    # "phantom member" test above stops firing -- but ensemble_fake still came from the ORIGINAL
    # scoring run, in which paas/pipeline.py built A2 alone. A fused column that equals one member's
    # column exactly IS that member, and the label must follow the data, not the config.
    if "ensemble_fake" in Xt:
        for _m in sorted(PER_MEMBER_COL):
            _c = PER_MEMBER_COL.get(_m)
            if _c and _c in Xt and np.array_equal(Xt[_c], Xt["ensemble_fake"]):
                print(f"[final] ensemble_fake is BIT-IDENTICAL to {_c} on test -- RELABELLING the "
                      f"fused column as the single member {_m}. It was written by a scoring run in "
                      f"which paas/pipeline.py built {_m} alone; the other members' columns come from "
                      f"score_9c_members.py and are NOT part of this fused score.")
                E9_LOADED.clear(); E9_LOADED[_m] = True
                break
    if E9_LOADED:
        _n = sorted(E9_LOADED)
        NICE["ensemble_fake"] = (f"9c ensemble = mean of {len(_n)} member(s): {', '.join(_n)}")
        print(f"[final] ensemble9 loaded {len(_n)} member(s): {', '.join(_n)}"
              + ("" if len(_n) > 1 else "   <-- a ONE-member 'ensemble'; the label says so on purpose"))
    else:
        print("[final] NOTE: score shards carry no ensemble9 membership; 'ensemble_fake' is labelled "
              "generically because the membership could not be verified.")

    rows = {}
    for k in members:
        r = {"model": NICE.get(k, k), "member_key": k,
             "test": {q: v for q, v in M.block(Xt[k], yt).items() if isinstance(v, (int, float))},
             "val": {q: v for q, v in M.block(Xv[k], yv).items() if isinstance(v, (int, float))},
             "applied_from_val": {}}
        for t in a.targets:
            tau, ach = M.tau_at_real_recall(Xv[k][yv == M.REAL], t)
            mt = Xt[k]
            blk = {"tau": float(tau),
                   "tau_source": f"FITTED ON es_val AT real_recall>={t:.2f} -- APPLIED TO es_test",
                   "real_rec_achieved_on_val": float(ach),
                   "real_rec_on_test": float((mt[yt == M.REAL] < tau).mean()),
                   "fake_rec_on_test": float((mt[yt != M.REAL] >= tau).mean()),
                   "pad_rec_on_test": float((mt[yt == M.PAD] >= tau).mean()),
                   "deepfake_rec_on_test": float((mt[yt == M.DEEPFAKE] >= tau).mean())}
            # binary accuracy at this operating point; the 3-class split of a fake needs per-model
            # class probabilities, which the ensemble members do not all expose, so it is not faked here
            pred_fake = mt >= tau
            blk["bin_acc_on_test"] = float((pred_fake == (yt != M.REAL)).mean())
            blk["bal_acc2_on_test"] = float(0.5 * (blk["real_rec_on_test"] + blk["fake_rec_on_test"]))
            r["applied_from_val"][f"real{int(t*100)}"] = blk

        # ---- SELF-FITTED matched-budget diagnostic ------------------------------------------
        # tau chosen ON es_test so that real recall == the target ON es_test.  This is NOT a
        # deployable number (it peeks at the test labels) and must NEVER be quoted beside an
        # applied_from_val figure.  Its only job is to make the models comparable at an EQUAL
        # real-recall cost: a val-fitted tau lands at a different achieved real recall for each
        # model, and two models at different real recall cannot be ranked on fake recall.
        r["self_fitted_on_test"] = {}
        for t in a.targets:
            tau_s, ach_s = M.tau_at_real_recall(Xt[k][yt == M.REAL], t)
            mt = Xt[k]
            r["self_fitted_on_test"][f"real{int(t*100)}"] = {
                "tau": float(tau_s),
                "tau_source": f"SELF-FITTED ON es_test AT real_recall>={t:.2f} -- NOT DEPLOYABLE",
                "real_rec_on_test": float(ach_s),
                "fake_rec_on_test": float((mt[yt != M.REAL] >= tau_s).mean()),
                "pad_rec_on_test": float((mt[yt == M.PAD] >= tau_s).mean()),
                "deepfake_rec_on_test": float((mt[yt == M.DEEPFAKE] >= tau_s).mean()),
            }
        r["views"] = {}
        for vname, drop in views.items():
            keep = np.array([p not in drop for p in tpaths]) if drop else np.ones(len(tpaths), bool)
            yk, xk = yt[keep], Xt[k][keep]
            vr = {"n": int(keep.sum()),
                  **{q: v for q, v in M.block(xk, yk).items() if isinstance(v, (int, float))}}
            tau, _ = M.tau_at_real_recall(Xv[k][yv == M.REAL], 0.98)
            vr["real_rec@val98"] = float((xk[yk == M.REAL] < tau).mean())
            vr["fake_rec@val98"] = float((xk[yk != M.REAL] >= tau).mean())
            r["views"][vname] = vr
        rows[k] = r
        print(f"[final] {NICE.get(k,k):<40} auc={r['test']['bin_auc']:.6f} ap={r['test']['ap']:.6f} "
              f"eer={r['test']['eer']:.6f} | @val-real98 real={r['applied_from_val']['real98']['real_rec_on_test']:.6f} "
              f"fake={r['applied_from_val']['real98']['fake_rec_on_test']:.6f}"
              f" | [self-fit@test-real98 fake={r['self_fitted_on_test']['real98']['fake_rec_on_test']:.6f}]")

    # ---- PE-SPC: scored from cached frozen features by train_pespc.py, on the SAME manifests ----
    pespc = None
    if os.path.exists(a.pespc):
        pespc = json.load(open(a.pespc))
        for nm, r in pespc["results"].items():
            key = f"pespc_{nm}"
            rows[key] = {"model": NICE.get(key, key), "member_key": key,
                         "test": {q: v for q, v in r["test"].items() if isinstance(v, (int, float))},
                         "val": {q: v for q, v in r["val"].items() if isinstance(v, (int, float))},
                         "applied_from_val": r["applied_from_val"],
                         "n_trainable": r.get("n_trainable"), "train_seconds": r.get("train_seconds")}
            sc = os.path.join(os.path.dirname(a.pespc), f"{nm}_test_scores.npy")
            if os.path.exists(sc):
                fs = np.load(sc)
                Pt = np.load(os.path.join(ES, "cache/es_test_0.npz"), allow_pickle=True)["paths"]
                Lt = np.load(os.path.join(ES, "cache/es_test_0.npz"), allow_pickle=True)["labels"]
                # matched-budget diagnostic, same caveat as the v4 members: NOT deployable,
                # never quoted beside an applied_from_val figure, exists only so PE-SPC and the
                # baselines can be ranked at an equal real-recall cost.
                yfull = Lt.astype(np.int64)
                rows[key]["self_fitted_on_test"] = {}
                for t in a.targets:
                    tau_s, ach_s = M.tau_at_real_recall(fs[yfull == M.REAL], t)
                    rows[key]["self_fitted_on_test"][f"real{int(t*100)}"] = {
                        "tau": float(tau_s),
                        "tau_source": f"SELF-FITTED ON es_test AT real_recall>={t:.2f} -- NOT DEPLOYABLE",
                        "real_rec_on_test": float(ach_s),
                        "fake_rec_on_test": float((fs[yfull != M.REAL] >= tau_s).mean()),
                        "pad_rec_on_test": float((fs[yfull == M.PAD] >= tau_s).mean()),
                        "deepfake_rec_on_test": float((fs[yfull == M.DEEPFAKE] >= tau_s).mean())}
                rows[key]["views"] = {}
                for vname, drop in views.items():
                    keep = np.array([p not in drop for p in Pt.tolist()]) if drop else np.ones(len(Pt), bool)
                    yk = Lt[keep].astype(np.int64)
                    rows[key]["views"][vname] = {
                        "n": int(keep.sum()),
                        **{q: v for q, v in M.block(fs[keep], yk).items()
                           if isinstance(v, (int, float))}}
            print(f"[final] {NICE.get(key,key):<40} auc={rows[key]['test']['bin_auc']:.6f} "
                  f"ap={rows[key]['test']['ap']:.6f} eer={rows[key]['test']['eer']:.6f}")
        # PE-SPC is scored on the FULL manifest while the v4 members are scored on the common-image
        # subset. Say so instead of implying identical denominators.
        if pespc["counts"]["es_test"] != len(tpaths):
            print(f"[final] NOTE: PE-SPC n={pespc['counts']['es_test']:,} vs v4-member common "
                  f"n={len(tpaths):,} (difference: images a v4 member could not read)")
    else:
        print(f"[final] NOTE: {a.pespc} absent -- PE-SPC rows omitted")

    # ---------------- FUSION: fitted on es_val, applied to es_test ----------------
    # Two fusions, because they answer different questions:
    #   like_for_like : the 4 retrained v4 members + PE-SPC on its FROZEN Exp-24 recipe. Every input
    #                   had the same privileges, so this is the honest "best achievable" number.
    #   plus_tuned    : the same, with the val-tuned PE-SPC row swapped in. Reported separately for the
    #                   same reason the starred rows are: its PE-SPC input got a privilege the
    #                   baselines did not.
    fusions = {}
    pes_dir = os.path.dirname(a.pespc)
    pes_paths_v = pes_paths_t = None
    if pespc:
        zv = np.load(os.path.join(ES, "cache/es_val_0.npz"), allow_pickle=True)
        zt = np.load(os.path.join(ES, "cache/es_test_0.npz"), allow_pickle=True)
        pes_paths_v, pes_paths_t = zv["paths"].tolist(), zt["paths"].tolist()
    for fname, pes_row in (("like_for_like", "default"), ("plus_tuned", "valsel_h4")):
        if not pespc or pes_row not in pespc.get("results", {}):
            continue
        fv = os.path.join(pes_dir, f"{pes_row}_val_scores.npy")
        ft = os.path.join(pes_dir, f"{pes_row}_test_scores.npy")
        if not (os.path.exists(fv) and os.path.exists(ft)):
            print(f"[final] fusion {fname}: {pes_row} score files missing -- skipped")
            continue
        pv = dict(zip(pes_paths_v, np.load(fv)))
        pt = dict(zip(pes_paths_t, np.load(ft)))
        vp = [p for p in vpaths if p in pv]
        tp = [p for p in tpaths if p in pt]
        if len(vp) < len(vpaths) or len(tp) < len(tpaths):
            print(f"[final] fusion {fname}: aligned {len(vp):,}/{len(vpaths):,} val and "
                  f"{len(tp):,}/{len(tpaths):,} test images across all members")
        vi = {p: k for k, p in enumerate(vpaths)}
        ti = {p: k for k, p in enumerate(tpaths)}
        Xv2 = {k: np.array([Xv[k][vi[p]] for p in vp]) for k in members}
        Xt2 = {k: np.array([Xt[k][ti[p]] for p in tp]) for k in members}
        Xv2["pespc"] = np.array([pv[p] for p in vp])
        Xt2["pespc"] = np.array([pt[p] for p in tp])
        yv2 = np.array([vlab[p] for p in vp]); yt2 = np.array([tlab[p] for p in tp])
        keys = members + ["pespc"]
        sv, st, wd = fuse(Xv2, yv2, Xt2, keys)
        tau, ach = M.tau_at_real_recall(sv[yv2 == M.REAL], 0.98)
        fusions[fname] = {
            "members": keys, "pespc_row": pes_row, "weights": wd,
            "n_val": len(vp), "n_test": len(tp),
            "test": {q: v for q, v in M.block(st, yt2).items() if isinstance(v, (int, float))},
            "applied_from_val": {"real98": {
                "tau": float(tau), "real_rec_achieved_on_val": float(ach),
                "real_rec_on_test": float((st[yt2 == M.REAL] < tau).mean()),
                "fake_rec_on_test": float((st[yt2 != M.REAL] >= tau).mean()),
                "pad_rec_on_test": float((st[yt2 == M.PAD] >= tau).mean()),
                "deepfake_rec_on_test": float((st[yt2 == M.DEEPFAKE] >= tau).mean()),
                "tau_source": "logistic fusion FITTED ON es_val -- applied unchanged to es_test"}}}
        # same matched-budget diagnostic as the members, so fusion-vs-member is compared at an
        # equal real-recall cost.  The FUSION WEIGHTS stay val-fitted; only tau is self-fitted.
        tau_s, ach_s = M.tau_at_real_recall(st[yt2 == M.REAL], 0.98)
        fusions[fname]["self_fitted_on_test"] = {"real98": {
            "tau": float(tau_s),
            "tau_source": "weights fitted on es_val; TAU SELF-FITTED ON es_test -- NOT DEPLOYABLE",
            "real_rec_on_test": float(ach_s),
            "fake_rec_on_test": float((st[yt2 != M.REAL] >= tau_s).mean()),
            "pad_rec_on_test": float((st[yt2 == M.PAD] >= tau_s).mean()),
            "deepfake_rec_on_test": float((st[yt2 == M.DEEPFAKE] >= tau_s).mean())}}
        print(f"[final] FUSION {fname:<14} auc={fusions[fname]['test']['bin_auc']:.6f} "
              f"| @val-real98 real={fusions[fname]['applied_from_val']['real98']['real_rec_on_test']:.6f} "
              f"fake={fusions[fname]['applied_from_val']['real98']['fake_rec_on_test']:.6f} "
              f"| [self-fit@test-real98 fake={fusions[fname]['self_fitted_on_test']['real98']['fake_rec_on_test']:.6f}] "
              f"| weights={ {k: round(v, 3) for k, v in wd.items()} }")

    leak = os.path.join(ES, "runs/old_eval_leak.json")
    out = {"ensemble9_members_loaded": sorted(E9_LOADED),
           "trainset": os.path.join(ES, "dataset/trainset.json"),
           "testset": os.path.join(ES, "dataset/testset.json"),
           "n_val": len(vpaths), "n_test": len(tpaths),
           "val_class_hist": np.bincount(yv, minlength=3).tolist(),
           "test_class_hist": np.bincount(yt, minlength=3).tolist(),
           "dropped_val": vdrop, "dropped_test": tdrop,
           "unreadable_val": vunread, "unreadable_test": tunread,
           "pespc_n": (pespc or {}).get("counts"),
           "old_eval_leak": json.load(open(leak)) if os.path.exists(leak) else None,
           "targets": a.targets, "rows": rows, "fusions": fusions,
           "key_semantics": {
               "bin_auc/ap/eer": "threshold-free, on es_test",
               "applied_from_val.*": "tau fitted on es_val, applied unchanged to es_test -- deployable",
               "fake_score": "1 - p(real) for PE-SPC; each v4 member's own fake score otherwise",
               "alignment": "members aligned by image path; only images all members scored are used",
               "views.full": "the whole clean testset",
               "views.no_oldsel_neardup": "minus images with PE cosine >= 0.98 to the OLD baseline "
                                          "selection set (mids_testset.json), which only the baselines' "
                                          "recipes ever saw"}}
    nan_rows = [k for k, v in rows.items()
                if not np.isfinite(v.get("test", {}).get("bin_auc", float("nan")))]
    if nan_rows:
        raise SystemExit(f"[final] REFUSING: bin_auc is not finite for {nan_rows}. A comparison table "
                         f"with a NaN headline metric must not be written.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"[final] -> {a.out}")


if __name__ == "__main__":
    main()
