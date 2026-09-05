#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Rebuild the retrain's ensemble9 config to list ALL THREE retrained 9-class members.

Why this exists: make_deploy_config.py:56 writes
    e9["members"] = [{"name": "A2_9c", "checkpoint": ...}]
i.e. the retrained deploy runs a ONE-member "ensemble", because v4 only ever retrained A2. With A1
and A3 retrained on the same EVAL_SPACE splits, the deployed config can carry all three, and
score_final.py can then emit per-member columns for a real combination search.

Every checkpoint must live under EVAL_SPACE: mixing a June-vintage A1/A3 (trained on the
undeduplicated data that overlaps this testset) into the comparison is exactly the contamination
this whole directory exists to remove, so it is refused rather than warned about.
"""
import json, os, shutil, sys

ES = "/datasets/work/vLLM/temp/EVAL_SPACE"
V4 = "/datasets/work/vLLM/temp/PAAS_ensemble_v4"
RUN = os.path.join(ES, "runs/retrain")
DEP = os.path.join(RUN, "deploy")

SRC = {"A1_9c": os.path.join(RUN, "a1_9c", "best.pt"),
       "A2_9c": os.path.join(RUN, "ninec", "best.pt"),
       "A3_9c": os.path.join(RUN, "a3_9c", "best.pt")}

def main():
    only = sys.argv[1:] or list(SRC)
    missing = [k for k in only if not os.path.exists(SRC[k])]
    if missing:
        raise SystemExit(f"[e9] REFUSING: no retrained checkpoint for {missing}\n"
                         + "\n".join(f"    expected {SRC[k]}" for k in missing)
                         + "\n    Has chain3_a1a3.sh finished? Do NOT substitute the deployed June "
                           "weights: they were trained on data that overlaps this testset.")
    if not os.path.isdir(DEP):
        raise SystemExit(f"[e9] REFUSING: {DEP} does not exist -- the retrain's deploy stage has not run.")

    # START FROM THE DEPLOYED CONFIG, not the template. $V4/config/ensemble9.json carries RELATIVE
    # base_models paths ("../base_models/t5-base") that only resolve from $V4/config/; copying them
    # into EVAL_SPACE/runs/retrain/deploy/ makes them point at runs/retrain/base_models, which does
    # not exist. make_deploy_config.py already resolved those to absolute paths in the deployed
    # config, so that is the correct base -- only `members` needs replacing.
    dep_e9 = os.path.join(DEP, "ensemble9.json")
    if os.path.exists(dep_e9):
        e9 = json.load(open(dep_e9))
        base_from = dep_e9
    else:
        tmpl = os.path.join(V4, "config", "ensemble9.json")
        e9 = json.load(open(tmpl))
        base_from = tmpl
        # resolve the template's relative paths against the TEMPLATE's directory, not the output's
        tdir = os.path.dirname(os.path.abspath(tmpl))
        for _k, _v in list((e9.get("base_models") or {}).items()):
            if _v and not os.path.isabs(_v):
                e9["base_models"][_k] = os.path.abspath(os.path.join(tdir, _v))
        print(f"[e9] NOTE: {dep_e9} absent; built from the template and resolved its relative "
              f"base_models paths against {tdir}")
    members = []
    for k in only:
        dst = os.path.join(DEP, f"{k}.pt")
        if os.path.abspath(SRC[k]) != os.path.abspath(dst):
            shutil.copy2(SRC[k], dst)
        if not os.path.abspath(dst).startswith(os.path.abspath(ES)):
            raise SystemExit(f"[e9] REFUSING: {dst} is outside EVAL_SPACE")
        members.append({"name": k, "checkpoint": dst})
        print(f"[e9] {k:<6} <- {SRC[k]}  ({os.path.getsize(dst)/1e6:.0f} MB)")
    e9["members"] = members
    e9["_about"] = (f"EVAL_SPACE retrain: all {len(members)} 9-class members retrained on the same "
                    f"deduplicated trainset and selected on the same val split. Replaces the "
                    f"single-member A2-only config that make_deploy_config.py writes.")
    out = os.path.join(DEP, "ensemble9_retrained_all.json")
    json.dump(e9, open(out, "w"), indent=2)
    print(f"[e9] -> {out}")

    # Point a COPY of the paas config at it. The original paas_retrained.json is left untouched so the
    # A2-only result stays reproducible and the two can be scored side by side.
    src_cfg = os.path.join(DEP, "paas_retrained.json")
    d = json.load(open(src_cfg))
    d.setdefault("ensemble9", {})["config_path"] = out
    comps = list((d.get("fusion") or {}).get("components") or [])
    # The fusion component list is deliberately NOT touched: it names the top-level PAAS fusion
    # (ffaa + A2_9c + gsd + selop), which is a different thing from the ensemble9 membership. The
    # search sees A1/A3 because score_final.py reads per-member columns straight from
    # ensemble_per_model, and `ensemble_fake` becomes the mean of THREE members automatically,
    # because the ensemble9 runtime averages whatever `members` lists.
    d["_about"] = "paas_retrained.json with the 3-member retrained ensemble9 (see ensemble9_retrained_all.json)"
    out_cfg = os.path.join(DEP, "paas_retrained_all9c.json")
    json.dump(d, open(out_cfg, "w"), indent=2)
    print(f"[e9] -> {out_cfg}   (fusion components unchanged: {comps})")
    # VALIDATE EVERY PATH IN WHAT WAS JUST WRITTEN. A config whose base_models do not resolve fails
    # only when the scoring job builds its first model -- 2.4 GPU-hours into the run, or worse, after
    # the first shard has already written partial output.
    bad = []
    def _check(label, val):
        # Only things that actually look like paths. Documentation fields (_about, _note, ...) are
        # prose that happens to contain slashes; treating them as paths made the first version of
        # this guard refuse a perfectly good config.
        if label.split(".")[-1].startswith("_") or label.startswith("_"):
            return
        if isinstance(val, str) and ("/" in val) and " " not in val and not val.startswith("cuda"):
            cand = val if os.path.isabs(val) else os.path.join(DEP, val)
            if not os.path.exists(cand):
                bad.append(f"{label}: {val}  ->  {os.path.abspath(cand)}")
    for _k, _v in (e9.get("base_models") or {}).items():
        _check(f"ensemble9.base_models.{_k}", _v)
    for _m in e9.get("members", []):
        _check(f"ensemble9.members[{_m.get('name')}].checkpoint", _m.get("checkpoint"))
    for _k, _v in d.items():
        if isinstance(_v, dict):
            for _k2, _v2 in _v.items():
                _check(f"{_k}.{_k2}", _v2)
        else:
            _check(_k, _v)
    if bad:
        raise SystemExit("[e9] REFUSING to hand over a config with unresolvable paths:\n    "
                         + "\n    ".join(bad)
                         + f"\n    (base config was {base_from})")
    print(f"[e9] every path in both configs resolves (base config: {os.path.basename(base_from)})")
    print("[e9] score with:  --config " + out_cfg)

if __name__ == "__main__":
    main()
