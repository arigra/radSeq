# Demo

Headings are plain text. Teal blocks are real `.py` files: typing in them edits the file on disk. Purple blocks run in a Jupyter kernel that starts in the project root.

```python
# the demo files live under .regain/demo, so put that on the path
import sys
sys.path.insert(0, ".regain/demo")
```

## 1 · Data
Make one fake Range-Doppler frame and turn it into a dB map.

```file .regain/demo/demo_src/data.py visual=visuals/data.svg
```

```python
from demo_src.data import fake_frame, power_map

m = power_map(fake_frame())
print(m.shape, "peak at", divmod(m.argmax(), m.shape[1]), f"{m.max():.1f} dB")
```

```python
import matplotlib.pyplot as plt

plt.imshow(m.T, aspect="auto", origin="lower")
plt.xlabel("range bin"); plt.ylabel("Doppler bin"); plt.colorbar(label="dB")
plt.show()
```

## 2 · Training
Add noise to a map and learn one weight that removes it.

```file .regain/demo/demo_src/train.py
```

```python
from demo_src.train import train

w, losses = train(m)
print(f"w = {w:.3f}")
```
