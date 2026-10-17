import json
from pathlib import Path

RUNS_DIR = Path("evaluation/runs")
REPORTS_DIR = Path("evaluation/reports")

def fix_sequences():
    # 1. Saari json files uthao
    files = list(RUNS_DIR.glob("*.json"))
    if not files:
        print("No run files found.")
        return

    # 2. Unhe date/time ke hisaab se sort karo
    runs_data = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            runs_data.append((f, data))
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    runs_data.sort(key=lambda x: x[1].get("created_at", ""))

    # 3. Sequentially rename karo
    for i, (filepath, data) in enumerate(runs_data, start=1):
        old_id = data.get("run_id", filepath.stem)
        new_id = f"run_{i:03d}"
        
        if old_id == new_id:
            print(f"[{new_id}] Already correct.")
            continue

        # Naya JSON data update karo
        data["run_id"] = new_id
        new_json_path = RUNS_DIR / f"{new_id}.json"
        
        # File save karo
        new_json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        # Purani file delete karo (kyunki rename ho gaya)
        if filepath.exists() and filepath != new_json_path:
            filepath.unlink()
            
        # Report file bhi rename karo
        old_md = REPORTS_DIR / f"{old_id}.md"
        new_md = REPORTS_DIR / f"{new_id}.md"
        if old_md.exists():
            content = old_md.read_text(encoding="utf-8")
            content = content.replace(old_id, new_id)  # Report ke andar ka naam bhi update karo
            new_md.write_text(content, encoding="utf-8")
            old_md.unlink()
            
        print(f"Renamed {old_id} -> {new_id}")

if __name__ == "__main__":
    fix_sequences()
    print("Done! All runs are now sequentially numbered.")