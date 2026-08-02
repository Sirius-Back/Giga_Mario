"""Compare plain MLP-VAE vs VGAE (GCN/GAT/SAGE) on hard L_hom + SD balance."""
from __future__ import annotations

import json
from pathlib import Path

from src.run.run_id.eval_vgae_legacy_losses import eval_run

ROOT = Path(__file__).resolve().parents[3]
OUT_JSON = ROOT / "VGAE" / "vae_vgae_architecture_comparison.json"
OUT_MD = ROOT / "VGAE" / "vae_vgae_architecture_comparison.md"


def _family(run: str) -> str:
    if run.startswith("mlp_vae") or run.startswith("VAE"):
        return "MLP-VAE (plain)"
    if "gcl_gat" in run:
        return "VGAE-GCL-GAT"
    if "gcl" in run:
        return "VGAE-GCL"
    if "appnp" in run:
        return "VGAE-APPNP"
    if "gcnii" in run:
        return "VGAE-GCNII"
    if "multik" in run:
        return "VGAE-GCN-multik"
    if "structfeat" in run:
        return "VGAE-GCN-struct"
    if "gat" in run:
        return "VGAE-GAT"
    if "sage" in run:
        return "VGAE-SAGE"
    if run.startswith("stage"):
        return "VGAE-GCN"
    return "other"


def main() -> int:
    rows = []
    for base in (ROOT / "VGAE", ROOT / "VAE"):
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if not p.is_dir():
                continue
            if not (p / "split.csv").is_file():
                continue
            # skip check dirs
            if p.name in {"checks"}:
                continue
            r = eval_run(p)
            if r.get("status") != "COMPLETED":
                continue
            meta = {}
            mp = p / "train_meta.json"
            if mp.is_file():
                meta = json.loads(mp.read_text(encoding="utf-8"))
            a = r["all_aggs"]
            o = r["sd_balance"]["ortho"]
            rows.append(
                {
                    "run": p.name,
                    "family": _family(p.name),
                    "path": str(p.relative_to(ROOT)),
                    "architecture": meta.get("architecture")
                    or ("mlp" if "mlp_vae" in p.name else "gcn"),
                    "loss_mode": r.get("loss_mode") or meta.get("loss_mode"),
                    "k": meta.get("k"),
                    "reported_best_l_hom": r.get("reported_best_l_hom")
                    or meta.get("best_l_hom"),
                    "legacy_mean": a["mean"]["l_hom"],
                    "weighted": a["weighted"]["l_hom"],
                    "robust": a["robust"]["l_hom"],
                    "log_balance": a["log_balance"]["l_hom"],
                    "ortho_p90_over_median": o["p90_over_median"],
                    "ortho_top5pct_mass": o["top5pct_mass_frac"],
                    "n_regions": r.get("n_regions"),
                }
            )

    rows.sort(key=lambda r: (r["family"], r.get("k") or 0, r["run"]))
    OUT_JSON.write_text(json.dumps({"models": rows}, indent=2) + "\n", encoding="utf-8")

    md = [
        "# MLP-VAE vs VGAE architecture comparison",
        "",
        "Hard metrics rescored from each run's best `split.csv` "
        "(same homology table / aggregators). "
        "Lower is better for mean / weighted / robust; "
        "log_balance is `E[log10 sd_ortho]−E[log10 sd_para]`.",
        "",
        "| Family | Run | Arch | k | Loss mode | Reported best | Legacy mean | Weighted | Robust | Log-bal | Ortho p90/med |",
        "|--------|-----|------|---|-----------|---------------|-------------|----------|--------|---------|---------------|",
    ]
    for r in rows:
        md.append(
            f"| {r['family']} | `{r['run']}` | {r['architecture']} | {r.get('k')} | "
            f"{r.get('loss_mode')} | "
            f"{r['reported_best_l_hom']:.4f} | {r['legacy_mean']:.4f} | "
            f"{r['weighted']:.4f} | {r['robust']:.4f} | {r['log_balance']:.4f} | "
            f"{r['ortho_p90_over_median']:.2f} |"
        )
    # Highlight bests among completed
    if rows:
        best_legacy = min(rows, key=lambda r: r["legacy_mean"])
        best_w = min(rows, key=lambda r: r["weighted"])
        md += [
            "",
            "## Highlights",
            "",
            f"- Best **legacy mean** L_hom: `{best_legacy['run']}` = "
            f"**{best_legacy['legacy_mean']:.4f}** ({best_legacy['family']})",
            f"- Best **weighted** L_hom: `{best_w['run']}` = "
            f"**{best_w['weighted']:.4f}** ({best_w['family']})",
            "",
            "Plain MLP-VAE numbers come from "
            "[VAE transcript](f7bb72bc-3b71-427b-be18-a2e42efb9a60) runs under `VAE/`.",
            "GAT/SAGE rows appear once those Stage1 trains finish.",
            "",
        ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(OUT_MD.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
