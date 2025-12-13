import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "seed_manifest.json"

def load_manifest():
    with open(MANIFEST, "r") as f:
        return json.load(f)

def scan_structure():
    m = load_manifest()
    expected = m["modules"]
    errors = []

    for module, subfolders in expected.items():
        base = ROOT / module
        if not base.exists():
            errors.append(f"[MISSING MODULE] {module}")
            continue

        for s in subfolders:
            if not (base / s).exists():
                errors.append(f"[MISSING SUBFOLDER] {module}/{s}")

    return errors

def drift_check():
    errors = scan_structure()
    if not errors:
        return {"status": "OK", "drift": False}

    return {
        "status": "DRIFT DETECTED",
        "drift": True,
        "errors": errors
    }

if __name__ == "__main__":
    print(drift_check())
