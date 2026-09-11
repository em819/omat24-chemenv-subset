#!/usr/bin/env python3
"""Turn MACE's statistics.json into the statistics_fixed.json stages 07 and 08 read.

Run it from the directory holding statistics.json; submit_preprocess.sh does
this automatically as its last step.

It makes three changes:

 1. Strips the np.float64(...) wrappers so atomic_energies is a plain literal.
 2. Keeps only the elements listed in atomic_numbers. Preprocessing writes E0s
    for more elements than the model uses, and the mismatch makes training fail
    at the first forward pass.
 3. Sets mean and std to 0.0 and 1.0, matching train.yaml and compile.yaml.
"""

import ast
import json
import re

# 0.0/1.0 means "do not rescale the energies": the loss works on raw energies
# and the E0s already remove the per-atom baseline.
FORCED_MEAN = 0.0
FORCED_STD = 1.0

with open("statistics.json") as f:
    data = json.load(f)

data["atomic_energies"] = re.sub(
    r"np\.float64\(([^)]+)\)", r"\1", data["atomic_energies"]
)

# Keep the elements the model was built for, in the same order.
zs = ast.literal_eval(data["atomic_numbers"])
ae = ast.literal_eval(data["atomic_energies"])
missing = [z for z in zs if z not in ae]
assert not missing, f"E0s missing for atomic numbers {missing}"
data["atomic_energies"] = str({int(z): ae[z] for z in zs})

print(f"mean: {data.get('mean')} -> {FORCED_MEAN}")
print(f"std : {data.get('std')} -> {FORCED_STD}")
data["mean"] = FORCED_MEAN
data["std"] = FORCED_STD

with open("statistics_fixed.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"wrote statistics_fixed.json ({len(zs)} elements)")
