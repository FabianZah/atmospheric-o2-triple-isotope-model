"""Check exact production Python pins against the public OSV advisory database."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())


def audit(lock: Path) -> dict:
    pins = [line.split("==") for line in lock.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    if any(len(row) != 2 for row in pins):
        raise ValueError("The runtime lock must contain exact name==version pins")
    queries = [{"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
               for name, version in pins]
    request = Request("https://api.osv.dev/v1/querybatch",
                      data=json.dumps({"queries": queries}).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        results = json.load(response)["results"]
    if len(results) != len(pins):
        raise ValueError("OSV returned an incomplete response")
    rows = [{"name": name, "version": version,
             "advisories": result.get("vulns", [])}
            for (name, version), result in zip(pins, results, strict=True)]
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "https://api.osv.dev/v1/querybatch", "packages": rows,
            "affected_package_count": sum(bool(row["advisories"]) for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/runtime_dependencies.json")
    args = parser.parse_args()
    report = audit(ROOT / "code/requirements-api-lock.txt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['affected_package_count']} affected packages; {args.output}")
    raise SystemExit(1 if report["affected_package_count"] else 0)


if __name__ == "__main__":
    main()
