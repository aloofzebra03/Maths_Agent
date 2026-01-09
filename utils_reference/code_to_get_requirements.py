import json, sys

fname = "pkgs.json"
try:
    # 'utf-8-sig' will silently drop a BOM if present
    with open(fname, encoding="utf-8-sig") as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    sys.exit(f"ERROR: couldn’t parse {fname} as JSON: {e}")

reqs = []
for pkg in data:
    n, v = pkg["name"], pkg["version"]
    if n.lower() in ("pip", "setuptools", "wheel"):
        continue
    # reqs.append(f"{n}=={v}")
    reqs.append(f"{n}")  # use >= to allow minor updates

with open("requirements.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(sorted(reqs)))

print("✔ wrote requirements.txt")


# pip list --format=json | Out-File pkgs.json -Encoding utf8


