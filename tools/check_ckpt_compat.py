import os
import sys

import torch

# Ensure project root is on sys.path so `import metai` works when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ckpt_path = os.environ.get("CKPT", "output/simvp/epoch=13-val_score=0.1147.ckpt")
    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("ckpt:", ckpt_path)
    if not os.path.exists(ckpt_path):
        print("ERROR: ckpt not found")
        return 2

    try:
        import mamba_ssm  # noqa: F401
        print("mamba_ssm: OK")
    except Exception as e:
        print("mamba_ssm: FAIL", repr(e))
        return 3

    from metai.model.simvp.simvp_trainer import SimVP as SimVP_S
    # Use the migrated, structure-preserving copy under mambax.
    from metai.model.mambax.trainer import SimVP as SimVP_M

    print("loading simvp ckpt...")
    simvp = SimVP_S.load_from_checkpoint(ckpt_path, map_location="cpu")
    print("loading mamba ckpt...")
    mamba = SimVP_M.load_from_checkpoint(ckpt_path, map_location="cpu")

    ks = set(simvp.state_dict().keys())
    km = set(mamba.state_dict().keys())
    print("keys simvp:", len(ks))
    print("keys mamba:", len(km))
    print("diff simvp-mamba:", len(ks - km))
    print("diff mamba-simvp:", len(km - ks))

    print("strict load simvp->mamba...")
    res = mamba.load_state_dict(simvp.state_dict(), strict=True)
    print("strict result:", res)

    simvp.eval()
    mamba.eval()
    with torch.no_grad():
        h = simvp.hparams
        in_shape = h.get("in_shape") if isinstance(h, dict) else getattr(h, "in_shape", None)
        if in_shape is None:
            in_shape = (10, 54, 64, 64)
        T, C, H, W = in_shape
        x = torch.randn(2, T, C, H, W)
        y_s = simvp(x)
        y_m = mamba(x)
    print("simvp out:", tuple(y_s.shape))
    print("mamba out:", tuple(y_m.shape))
    print("max abs diff:", (y_s - y_m).abs().max().item())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


