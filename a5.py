import json, pandas as pd, re

# --- load inputs ---
spec_stream = json.load(open("streaming.json"))
spec_backfill = json.load(open("backfill.json"))
df = pd.read_csv("merged.csv", nrows=0)

# --- pull requested columns from both specs ---
def pull_cols(spec):
    # adjust keys if your spec nests by entity
    cols = []
    for k in ("crashes","vehicles","people"):
        if k in spec: cols += spec[k]
    return cols

requested = pull_cols(spec_stream) + pull_cols(spec_backfill)

# --- normalize helpers ---
norm = lambda s: re.sub(r"\s+"," ", str(s).strip().lower())

req_norm = [norm(c) for c in requested]
csv_norm = [norm(c) for c in df.columns]

# optional: seed known aliases here (left=spec name, right=csv name)
aliases = {
    "driver_id":"person_id",
    "veh_no":"vehicle_number",
    # add more as we find them
}

# apply aliases to requested side
req_norm_alias = [norm(aliases.get(c, c)) for c in req_norm]

# --- coverage check ---
missing = sorted(set(req_norm_alias) - set(csv_norm))
extra   = sorted(set(csv_norm) - set(req_norm_alias))

print("Missing from merged.csv:", missing)
print("Extra in merged.csv:", extra)
