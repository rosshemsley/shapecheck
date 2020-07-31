# 🔸 Shapecheck

shapecheck is a library to help you ensure tensors are correctly sized at runtime.

```python
import shapecheck

@shapecheck.check_args(x=(1,3,3,4,5), img=("N", ("R", "G", "B"), "H", "W"))
def f(x, img):
    ...

x = torch.rand((1, 3, 3, 4, 5))
img = torch.rand((1, 3, 256, 256))
f(x, img=img)

```
