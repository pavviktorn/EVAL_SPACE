#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Build PE-SPC class prototypes from the PE-Core-L14-336 TEXT tower.

Mirrors PAAS_simplicity/spc/text_prototypes.py, with three things that MUST differ for L14 and that
would silently corrupt the prototypes if copied over from the G14 script:

  1. output_dim is 1024, not 1280. The head's `dim` must match or the matmul shape-errors (loudly)
     -- or worse, a stale 1280-d prototype file loads into a 1024-d head and asserts.
  2. context_length is 32, not 72. The longest prompts in this project's sets are close to that, and
     a tokenizer TRUNCATES SILENTLY. This script therefore round-trips every prompt through the
     tokenizer and REFUSES if any prompt loses tokens -- a truncated prompt is a different prompt,
     and the resulting prototype would be wrong in a way nothing downstream could detect.
  3. The prompt STRINGS are imported from the G14 module rather than retyped, so the two encoders are
     compared on identical wording. The only difference between the two prototype files is the tower
     that embedded them.

Output goes to EVAL_SPACE/cache/ (Phase D: EVAL_SPACE is the workspace).
"""
import ast, hashlib, os, sys
import torch
import torch.nn.functional as F

PS = "/datasets/work/vLLM/temp/PAAS_simplicity"
ES = "/datasets/work/vLLM/temp/EVAL_SPACE"
sys.path.insert(0, os.path.join(PS, "perception_models"))
import core.vision_encoder.pe as pe                      # noqa: E402
import core.vision_encoder.transforms as pt              # noqa: E402

MODEL = "PE-Core-L14-336"
CKPT = "/datasets/work/vLLM/temp/PE-Core-L14-336/PE-Core-L14-336.pt"

# Parse TRIPLETS and K4 out of the G14 module WITHOUT importing it (importing would load the G14
# checkpoint). Same strings, guaranteed.
_src = open(os.path.join(PS, "spc/text_prototypes.py")).read()
_tree = ast.parse(_src)
TRIPLETS, K4 = None, None
for _n in _tree.body:
    if isinstance(_n, ast.Assign):
        _t = getattr(_n.targets[0], "id", "")
        if _t == "TRIPLETS":
            TRIPLETS = ast.literal_eval(_n.value)
        elif _t == "K4":
            K4 = ast.literal_eval(_n.value)
assert TRIPLETS and K4, "could not parse TRIPLETS/K4 from text_prototypes.py"
print(f"[protoL14] parsed {len(TRIPLETS)} triplets and K4 with "
      f"{ {k: len(v) for k, v in K4.items()} } from the G14 module (identical wording)")


def main():
    dev = os.environ.get("PROTO_DEVICE", "cpu")
    model = pe.CLIP.from_config(MODEL, pretrained=True, checkpoint_path=CKPT).to(dev).eval()
    ctx = int(model.context_length)
    dim = int(model.text_projection.shape[-1])
    tok = pt.get_text_tokenizer(ctx)
    print(f"[protoL14] {MODEL} context_length={ctx} clip_dim={dim} "
          f"logit_scale.exp()={model.logit_scale.exp().item():.4f}")

    # ---- TRUNCATION GATE -------------------------------------------------------------------
    # A tokenizer that truncates returns a full-length row with no error. Detect it by checking
    # whether the EOT token survived: if the last non-pad slot is not EOT, the prompt was cut.
    every = [s for tri in TRIPLETS.values() for s in tri] + [s for v in K4.values() for s in v]
    t = tok(every)
    bad = []
    for s, row in zip(every, t):
        nz = (row != 0).nonzero().flatten()
        last = int(row[nz[-1]]) if len(nz) else -1
        # EOT is the highest special id emitted by this tokenizer for a complete sequence
        if len(nz) >= ctx and last != int(t[0][(t[0] != 0).nonzero().flatten()[-1]]):
            bad.append((s, len(nz)))
    longest = max((int((row != 0).sum()), s) for s, row in zip(every, t))
    print(f"[protoL14] longest prompt uses {longest[0]}/{ctx} tokens: {longest[1]!r}")
    if bad:
        raise SystemExit(f"[protoL14] REFUSING: {len(bad)} prompt(s) TRUNCATED at context_length="
                         f"{ctx}: {bad[:5]}. A truncated prompt is a different prompt.")
    if longest[0] >= ctx:
        raise SystemExit(f"[protoL14] REFUSING: a prompt fills the entire {ctx}-token context, which "
                         f"means it may have lost its EOT. Shorten the prompt set for this encoder.")

    @torch.no_grad()
    def emb(strs):
        return model.encode_text(tok(list(strs)).to(dev), normalize=True).float()

    out = {}
    for name, tri in TRIPLETS.items():
        e = emb(tri); out[name] = e
        g = e @ e.t()
        print(f"[protoL14] {name:<3} cos(real,pad)={g[0,1]:.4f} cos(real,df)={g[0,2]:.4f} "
              f"cos(pad,df)={g[1,2]:.4f}")
    stack = torch.stack([out[k] for k in TRIPLETS])
    out["T10"] = F.normalize(stack.mean(0), dim=-1)
    out["K4"] = torch.cat([emb(K4[c]) for c in ("real", "pad", "deepfake")], 0)
    g = out["K4"] @ out["K4"].t()
    print(f"[protoL14] K4 {tuple(out['K4'].shape)} (class-major: real0-3, pad4-7, deepfake8-11)")
    # separation WITHIN vs BETWEEN classes: if between >= within the init is collapsed and the head
    # starts with nothing to work with.
    import itertools
    win = [g[a, b].item() for c in range(3) for a, b in itertools.combinations(range(c*4, c*4+4), 2)]
    bet = [g[a, b].item() for ca in range(3) for cb in range(ca+1, 3)
           for a in range(ca*4, ca*4+4) for b in range(cb*4, cb*4+4)]
    print(f"[protoL14] K4 mean cos WITHIN class={sum(win)/len(win):.4f}  "
          f"BETWEEN classes={sum(bet)/len(bet):.4f}  (within should exceed between)")

    h = hashlib.sha256()
    with open(CKPT, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    fp = {"encoder_sha256": h.hexdigest(), "context_length": ctx, "clip_dim": dim,
          "model": MODEL, "tokenizer": "pt.get_text_tokenizer(model.context_length)"}
    dst = os.path.join(ES, "cache/prototypes_L14.pt")
    torch.save({"protos": out, "triplets": TRIPLETS, "k4": K4, "fingerprint": fp,
                "logit_scale_exp": float(model.logit_scale.detach().exp())}, dst)
    print(f"[protoL14] encoder_sha256={fp['encoder_sha256'][:16]}...  -> {dst}")


if __name__ == "__main__":
    main()
