# Chebyshev Distance Transform

This is a PyTorch extension that implements distance transform using the
Chebyshev (or chestboard) distance. It computes the shortest distance between
each foreground pixel and the background. Pixels valued -1 are considered
background and +1 foreground.

## How to use it

Provided PyTorch is present at the system, build (or even install) the
extension:
```shell
python setup.py build
# or
python setup.py install
```

If you do not wish to install the extension, adjust `PYTHONPATH` accordingly to
point at the extension build:
```shell
export PYTHONPATH="$PYTHONPATH:/some/path/to/chdt/build/lib.linux-x86_64-3.8/"
```

Then use it like this in your scripts:
```python
import torch # this import must come first
import chdt

Y = ... # binary {+-1}-valued tensor of shape (B, 1, H, W)
D = chdt.transform(Y)
if (D == chdt.INF).any():
    print("There were no background pixels")
```

## Correctness

The extension is tested to return the same results as the Scipy's
`distance_transform_cdt` function with one exception: `distance_transform_cdt`
assigns `-1` to all pixels if there is no background, whereas `chdt.transform`
assigns `chdt.INF`.

