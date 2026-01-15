import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

npz_path = Path("parsed_signals_full/1004800_6025_0_0.npz")
out_png = npz_path.with_suffix(".png")

z = np.load(npz_path)

fs = float(z["fs"][0])
t = np.arange(len(z["I"])) / fs

plt.figure(figsize=(16,8))

plt.subplot(3,1,1)
plt.plot(t, z["I"])
plt.ylabel("Lead I (mV)")
plt.title("Full Disclosure ECG – Entire Recording")

plt.subplot(3,1,2)
plt.plot(t, z["II"])
plt.ylabel("Lead II (mV)")

plt.subplot(3,1,3)
plt.plot(t, z["III"])
plt.ylabel("Lead III (mV)")
plt.xlabel("Time (seconds)")

plt.tight_layout()
plt.savefig(out_png, dpi=150)
plt.close()

print(f"Saved plot to: {out_png}")
