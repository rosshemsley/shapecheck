# 🔸 Shapecheck

shapecheck is a library to help you ensure tensors are correctly sized at runtime.

```
pip install git+https://github.com/rosshemsley/shapecheck
```
## How it works
Add a decorator to your classes taking multidimensional arrays with a `.shape` attribute.
For each argument you wish to shapecheck, add a label. This label is a tuple where each element is identified
one-to-one with an element in the shape.

The label tuple accepts multiple values. Strings are interpreted as dimension names, and allow the dimension to have any size.
Integer values give the size of the dimension, and tuples allow the naming of individual fields within the dimension.
For tuples, the shapechecker will ensure that the number of fields in the tuple matches the dimension size.

## Quickstart

```python
import shapecheck

@shapecheck.check_args(x=(1,3,3,4,5), img=("N", ("R", "G", "B"), "H", "W"))
def f(x, img):
    ...

x = torch.rand((1, 3, 3, 4, 5))
img = torch.rand((1, 3, 256, 256))
f(x, img=img)

```
