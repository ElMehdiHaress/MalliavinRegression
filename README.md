# Malliavin Regression

![CI](https://github.com/ElMehdiHaress/MalliavinRegression/actions/workflows/ci.yml/badge.svg)

Library extracted from the `MalliavinDescent.ipynb` notebook. Core functions live in `src/malliavin_regression/`. Examples moved to `examples/`.

## 🚀 Quick start
```bash
pip install "git+https://github.com/ElMehdiHaress/MalliavinRegression.git"
import malliavin_regression as mr
d = mr.generate_data(50, 20, noise_var=1, name_model="sinus", show=False)
print(type(d))
```
## Example
python examples/demo.py

## Layout
src/malliavin_regression/   # library code (no top-level execution)

examples/                   # runnable demos

tests/                      # minimal pytest

## License
MIT


