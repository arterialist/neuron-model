"""Report the retained plasticity campaign records without assumed completions."""
import argparse
import json
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=repo / "foveation_results/minibrain/arch")
    args = parser.parse_args()
    records = []
    unreadable = 0
    for path in sorted(args.input_root.expanduser().glob("plast_*_seed0.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                record = json.load(stream)
            if not isinstance(record, dict):
                raise ValueError("expected an object")
            records.append((path.stem, record))
        except (OSError, ValueError):
            unreadable += 1
    done = sum(bool(record.get("complete", False)) for _, record in records)
    running = []
    for stem, record in records:
        if record.get("complete", False):
            continue
        ticks = record.get("traj_t", [])
        last = ticks[-1] if isinstance(ticks, list) and ticks else 0
        running.append(f"{record.get('tag', stem)}@{last}")
    tail = ", ".join(running) or "none in retained records"
    print(f"retained plasticity records: {done}/{len(records)} complete | running: {tail} | unreadable: {unreadable}")


if __name__ == "__main__":
    main()
