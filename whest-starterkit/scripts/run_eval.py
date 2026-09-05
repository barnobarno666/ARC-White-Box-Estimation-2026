import hashlib
import json
import subprocess
import sys
from pathlib import Path

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_eval(estimator_path: str, variant_name: str, desc: str, update_champion: bool = False):
    file_hash = compute_sha256(estimator_path)
    cmd = [
        "uv", "run", "whest", "run",
        "--estimator", estimator_path,
        "--dataset", r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini",
        "--split", "mini",
        "--n-mlps", "8",
        "--runner", "local",
        "--format", "json"
    ]
    print(f"Running evaluation on {estimator_path} (SHA256: {file_hash[:16]}...)...")
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**dict(subprocess.os.environ), "PYTHONUTF8": "1"})
    if proc.returncode != 0:
        print("ERROR running whest:")
        print(proc.stderr)
        return None

    stdout = proc.stdout
    json_start = stdout.find('{')
    if json_start == -1:
        print("Could not find JSON output:")
        print(stdout)
        return None

    data = json.loads(stdout[json_start:])
    results = data["results"]

    adj_score = results["adjusted_final_layer_score"]
    raw_mse = results["final_layer_mse"]
    mult = results["mean_score_multiplier"]
    failures = results["n_failed_mlps"]
    util = results["mean_compute_utilization"]
    
    per_mlp = results["per_mlp"]
    max_res_time = max(m.get("residual_wall_time_s", 0) for m in per_mlp)

    # Save as new champion if requested
    if update_champion:
        champ_file = Path("scripts/champion_8mlp.json")
        with open(champ_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Updated {champ_file} with current results (Adjusted Score: {adj_score:.6e}).")

    # Load baseline
    with open("scripts/baseline_8mlp.json", "r") as f:
        baseline = json.load(f)
    if "results" in baseline:
        baseline = baseline["results"]
    base_per_mlp = baseline["per_mlp"]

    # Load Phase 4 Control / Champion
    champion = None
    champ_per_mlp = None
    if Path("scripts/champion_8mlp.json").exists():
        with open("scripts/champion_8mlp.json", "r") as f:
            champion = json.load(f)
        if "results" in champion:
            champion = champion["results"]
        champ_per_mlp = champion["per_mlp"]

    wins_base = losses_base = ties_base = 0
    wins_champ = losses_champ = ties_champ = 0
    max_regression_vs_champ = -999.0
    diffs_champ = []

    print("\n--- Per-MLP Head-to-Head Comparison ---")
    for i, m_cand in enumerate(per_mlp):
        name = m_cand.get("mlp_name", f"MLP_{i}")
        s_cand = m_cand.get("adjusted_final_layer_score", 0)
        s_base = base_per_mlp[i].get("adjusted_final_layer_score", 0)
        diff_base = ((s_cand - s_base) / s_base) * 100 if s_base else 0

        if s_cand < s_base:
            wins_base += 1
            res_base = "W"
        elif s_cand > s_base:
            losses_base += 1
            res_base = "L"
        else:
            ties_base += 1
            res_base = "T"

        champ_str = ""
        if champ_per_mlp is not None:
            s_champ = champ_per_mlp[i].get("adjusted_final_layer_score", 0)
            diff_champ = ((s_cand - s_champ) / s_champ) * 100 if s_champ else 0
            diffs_champ.append(diff_champ)
            if diff_champ > max_regression_vs_champ:
                max_regression_vs_champ = diff_champ

            if s_cand < s_champ:
                wins_champ += 1
                res_champ = "WIN"
            elif s_cand > s_champ:
                losses_champ += 1
                res_champ = "LOSS"
            else:
                ties_champ += 1
                res_champ = "TIE"
            champ_str = f" | Champ={s_champ:.4e} (Diff={diff_champ:+.2f}% -> {res_champ})"

        print(f"[{i+1}/8] {name:<22}: Cand={s_cand:.4e} | Base={s_base:.4e} ({res_base}){champ_str}")

    win_str_base = f"{wins_base}W-{losses_base}L" if ties_base == 0 else f"{wins_base}W-{losses_base}L-{ties_base}T"
    win_str_champ = f"{wins_champ}W-{losses_champ}L" if ties_champ == 0 else f"{wins_champ}W-{losses_champ}L-{ties_champ}T"

    champ_score = champion["adjusted_final_layer_score"] if champion else 1.223643e-07
    rel_diff_champ = ((adj_score - champ_score) / champ_score) * 100

    # Phase 4 Strict Promotion Gates:
    # 1. Primary Target: < 8.00e-8
    # 2. Stretch Target: < 4.00e-8
    # 3. Interim Promotable: >= 10% improvement (rel_diff_champ <= -10.0%), >= 6/8 wins, 0 failures, max_res_time < 0.35s, no regression > 10%
    if adj_score < 4.00e-8 and failures == 0:
        decision = "PHASE 4 STRETCH TARGET ACHIEVED (<4.00e-8)"
    elif adj_score < 8.00e-8 and failures == 0 and wins_champ >= 6:
        decision = "PHASE 4 PRIMARY TARGET ACHIEVED (<8.00e-8)"
    elif rel_diff_champ <= -10.0 and wins_champ >= 6 and failures == 0 and max_res_time < 0.35 and max_regression_vs_champ <= 10.0:
        decision = f"PROMOTABLE INTERIM CHAMPION ({win_str_champ} vs Champ, Diff={rel_diff_champ:+.2f}%)"
    elif adj_score < champ_score and failures == 0:
        decision = f"POTENTIAL ({win_str_champ} vs Champ, Diff={rel_diff_champ:+.2f}%)"
    elif adj_score < baseline["adjusted_final_layer_score"] and failures == 0:
        decision = f"REJECT ({win_str_champ} vs Champ, Diff={rel_diff_champ:+.2f}%)"
    else:
        decision = "REJECT (Regression)"

    print("\n=== SUMMARY ===")
    print(f"Variant:            {variant_name}")
    print(f"Candidate SHA256:   {file_hash}")
    print(f"Adjusted Score:     {adj_score:.6e} (Champ: {champ_score:.6e}, Diff: {rel_diff_champ:+.2f}%)")
    print(f"Raw Final MSE:      {raw_mse:.6e}")
    print(f"Mean Multiplier:    {mult:.4f} (Util: {util * 100:.2f}%)")
    print(f"Max Residual Time:  {max_res_time:.4f}s")
    print(f"Worst-MLP Diff:     {max_regression_vs_champ:+.2f}%")
    print(f"Failures:           {failures}")
    print(f"Head-to-head Champ: {win_str_champ}")
    print(f"Head-to-head Base:  {win_str_base}")
    print(f"Decision:           {decision}")

    table_row = f"| {variant_name} | {desc} | {raw_mse:.4e} | {mult:.4f} | {adj_score:.4e} | {rel_diff_champ:+.2f}% | {win_str_champ} | {max_res_time:.4f}s | {decision} |"
    print("\nMarkdown Table Row for Current thing.md:")
    print(table_row)
    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_eval.py <estimator_path> <variant_name> [desc] [--update-champion]")
        sys.exit(1)
    est = sys.argv[1]
    name = sys.argv[2]
    d = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "--update-champion" else ""
    up_champ = "--update-champion" in sys.argv
    run_eval(est, name, d, update_champion=up_champ)
