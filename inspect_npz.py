import numpy as np

p = "dataset_cz_v2.npz"
obj = np.load(p, allow_pickle=True)
print("keys:", obj.files)
for k in obj.files:
    a = obj[k]
    print(k, type(a), getattr(a, "shape", None), getattr(a, "dtype", None))

raw = obj["raw"]
clean = obj["clean"]
sec = obj["section_id"]

print("raw min/max/mean/std:", raw.min(), raw.max(), raw.mean(), raw.std())
print("clean min/max/mean/std:", clean.min(), clean.max(), clean.mean(), clean.std())

sec_unique = np.unique(sec)
print("section_id unique:", len(sec_unique), "first10:", sec[:10])
print("len raw:", len(raw), "len clean:", len(clean), "len section_id:", len(sec))
