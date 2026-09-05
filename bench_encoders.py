#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""GPU-BOUND throughput for both encoders, so the accuracy/cost trade is measured, not assumed.

The extraction rates from run_extract_l14.sh CANNOT be used for this: they were storage-bound (GPUs
sat at 0-83% util while a 300-file stat sample took over two minutes), so they measure the filesystem,
not the encoder. This feeds SYNTHETIC tensors already on the GPU -- no dataloader, no decode, no I/O --
which is the only way to isolate encoder cost.
"""
import sys, time
import torch

sys.path.insert(0, "/datasets/work/vLLM/temp/PAAS_simplicity/perception_models")
import core.vision_encoder.pe as pe                                     # noqa: E402

M = {"PE-Core-G14-448": "/datasets/work/vLLM/temp/PE-Core-G14-448/PE-Core-G14-448.pt",
     "PE-Core-L14-336": "/datasets/work/vLLM/temp/PE-Core-L14-336/PE-Core-L14-336.pt"}
BS = 64
print(f"{'encoder':<18}{'params(M)':>11}{'img_size':>10}{'dim':>7}{'img/s':>10}{'ms/img':>9}")
res = {}
for name, ckpt in M.items():
    model = pe.CLIP.from_config(name, pretrained=True, checkpoint_path=ckpt).cuda().eval()
    model = model.to(torch.bfloat16)
    vis = model.visual
    npar = sum(p.numel() for p in vis.parameters()) / 1e6
    sz = model.image_size
    x = torch.randn(BS, 3, sz, sz, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(3):                                   # warmup: cudnn autotune + graph capture
            model.encode_image(x, normalize=True)
        torch.cuda.synchronize()
        t0 = time.time()
        N = 12
        for _ in range(N):
            f = model.encode_image(x, normalize=True)
        torch.cuda.synchronize()
        el = time.time() - t0
    ips = BS * N / el
    res[name] = ips
    print(f"{name:<18}{npar:>11.1f}{sz:>10}{f.shape[-1]:>7}{ips:>10.1f}{1000/ips:>9.3f}")
    del model, x
    torch.cuda.empty_cache()
g, l = res["PE-Core-G14-448"], res["PE-Core-L14-336"]
print(f"\nL14-336 is {l/g:.2f}x the throughput of G14-448 "
      f"({g:.0f} -> {l:.0f} img/s on one GPU, batch {BS}, bf16, synthetic input)")
