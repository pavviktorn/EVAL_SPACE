#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Compare the four step-3/4/5 searches and answer one question: does PE-SPC help the fusion?

The four runs differ in exactly two variables, so differences are attributable:
    devsel_none    PE-SPC=devsel_h3   normalize=none     production-style mean
    devsel_cdfval  PE-SPC=devsel_h3   normalize=cdfval   val-fitted CDF  (DEPLOYABLE)
    devsel_rank    PE-SPC=devsel_h3   normalize=rank     per-split rank  (TRANSDUCTIVE, diagnostic)
    devbin_none    PE-SPC=devsel_bin  normalize=none     binary PE-SPC instead of 3-class

READING RULE, fixed before the numbers were opened: es_val is a DISCREDITED selector here
(runs/val_clean_selection.json: Spearman vs test = -0.76 in the top 12). So "the val-argmax's test
score" is reported as a DEPLOYABLE number -- what you would actually have shipped -- and NOT as
evidence about which fusion is best. Oracle rows are labelled ORACLE and are not attainable.
"""
import json
import numpy as np

RUNS = [("devsel_none", "3-class PE-SPC, mean, no norm"),
        ("devsel_cdfval", "3-class PE-SPC, val-CDF norm (deployable)"),
        ("devsel_rank", "3-class PE-SPC, rank norm (transductive)"),
        ("devbin_none", "BINARY PE-SPC, mean, no norm")]
D = {t: json.load(open(f"runs/comb_{t}.json")) for t, _ in RUNS}


def mean_rows(d):
    return {k[len("mean::"):]: v for k, v in d["rows"].items() if k.startswith("mean::")}


# ---------- 0. sanity: a monotone per-member transform cannot move a SINGLE-member AUC ----------
print("=" * 100)
print("SANITY CHECK -- normalisation is monotone per member, so single-member test AUC must not move")
base = mean_rows(D["devsel_none"])
singles = sorted([k for k in base if "+" not in k])
# TOLERANCES, and why they differ:
#   rank   must be EXACT. Tie-averaged ranks are a pure function of the score, strictly order-
#          preserving, so every single-member AUC must reproduce bit-for-bit. Any drift is a bug.
#   cdfval is allowed 1e-6. The val-fitted CDF is monotone but not STRICTLY monotone: two distinct
#          test scores with no val score between them collapse to one value. That is inherent to
#          fitting the map on 50,384 val rows and is the price of the transform being deployable.
TOL = {"cdfval": 1e-6, "rank": 0.0}
bad = []
for m in singles:
    a = base[m]["test"]["bin_auc"]
    b = mean_rows(D["devsel_cdfval"])[m]["test"]["bin_auc"]
    c = mean_rows(D["devsel_rank"])[m]["test"]["bin_auc"]
    ok_b, ok_c = abs(a - b) <= TOL["cdfval"], abs(a - c) <= TOL["rank"]
    if not ok_b: bad.append(f"{m}/cdfval")
    if not ok_c: bad.append(f"{m}/rank")
    print(f"  {m:<8} none={a:.9f} cdfval={b:.9f} (d={b-a:+.1e} {'ok' if ok_b else 'FAIL'}) "
          f"rank={c:.9f} ({'exact' if ok_c else 'FAIL'})")
print(f"  -> {'PASS -- rank exact on all 9, cdfval within its documented 1e-6' if not bad else 'FAIL: ' + ', '.join(bad)}")

# ---------- 1. per-run: what val picks, what it costs, what the oracle was ----------
print("\n" + "=" * 100)
print("WHAT EACH RUN WOULD HAVE SHIPPED (val-selected) vs WHAT WAS AVAILABLE (oracle, unattainable)")
print(f"{'run':<14} {'val-argmax combination':<34} {'its test':>10} {'ORACLE':>10} {'gap':>9} {'oracle combination'}")
summary = {}
for tag, _ in RUNS:
    R = mean_rows(D[tag])
    va = max(R, key=lambda k: R[k]["val"]["bin_auc"])
    orc = max(R, key=lambda k: R[k]["test"]["bin_auc"])
    va_t, orc_t = R[va]["test"]["bin_auc"], R[orc]["test"]["bin_auc"]
    summary[tag] = dict(val_argmax=va, val_argmax_test=va_t, oracle=orc, oracle_test=orc_t)
    print(f"{tag:<14} {va:<34} {va_t:>10.6f} {orc_t:>10.6f} {orc_t - va_t:>9.6f} {orc}")

# ---------- 2. the actual question: does adding PE-SPC to a fusion help ON TEST? ----------
print("\n" + "=" * 100)
print("DOES ADDING PE-SPC HELP?  every fusion, with vs without PE-SPC, paired on test")
for tag, desc in RUNS:
    R = mean_rows(D[tag])
    deltas = []
    for k in R:
        if "PE-SPC" in k:
            continue
        withp = "+".join(sorted(k.split("+") + ["PE-SPC"], key=lambda x: (x == "PE-SPC", x)))
        cand = [kk for kk in R if "PE-SPC" in kk and set(kk.split("+")) == set(k.split("+")) | {"PE-SPC"}]
        if cand:
            deltas.append((R[cand[0]]["test"]["bin_auc"] - R[k]["test"]["bin_auc"], k, cand[0]))
    deltas.sort()
    arr = np.array([d[0] for d in deltas])
    print(f"\n  [{tag}] {desc}")
    print(f"    {len(arr)} paired fusions | PE-SPC helps in {int((arr > 0).sum())} ({(arr > 0).mean():.1%}) "
          f"| median delta {np.median(arr):+.6f} | mean {arr.mean():+.6f}")
    print(f"    best  gain: {deltas[-1][0]:+.6f}  {deltas[-1][1]} -> +PE-SPC")
    print(f"    worst loss: {deltas[0][0]:+.6f}  {deltas[0][1]} -> +PE-SPC")

# ---------- 3. head-to-head on the recipes that matter ----------
print("\n" + "=" * 100)
print("KEY RECIPES, test bin_auc / test deepfake recall at the val-fitted 98%-real threshold")
WATCH = ["ffaa", "PE-SPC", "ffaa+PE-SPC", "ffaa+A1+A2+gsd+selop",
         "A1+PE-SPC+ffaa+gsd+selop", "ffaa+A1+A2+gsd+selop+PE-SPC",
         "ffaa+A2", "ffaa+A2+PE-SPC"]
hdr = f"{'recipe':<32}" + "".join(f"{t.replace('devsel_','').replace('_none',''):>13}" for t, _ in RUNS)
print(hdr)
for w in WATCH:
    line = f"{w:<32}"
    for tag, _ in RUNS:
        R = mean_rows(D[tag])
        k = next((kk for kk in R if set(kk.split("+")) == set(w.split("+"))), None)
        line += f"{R[k]['test']['bin_auc']:>13.6f}" if k else f"{'-':>13}"
    print(line)
print()
for w in WATCH:
    line = f"{w:<32}"
    for tag, _ in RUNS:
        R = mean_rows(D[tag])
        k = next((kk for kk in R if set(kk.split("+")) == set(w.split("+"))), None)
        v = R[k]["applied_from_val"]["real0.98"]["deepfake_rec_on_test"] if k else None
        line += f"{v:>13.6f}" if v is not None else f"{'-':>13}"
    print(line + ("   <- deepfake recall @ val-fitted tau" if w == WATCH[0] else ""))

json.dump(summary, open("runs/step345_summary.json", "w"), indent=2)
print("\n-> runs/step345_summary.json")
