import torch

import shapecheck


def test_1():
    v1 = torch.rand((1, 3, 4, 5))
    shapecheck.check(v1, ("N", ("R", "G", "B"), "H", "W"))

    CLASSES = ("background", "car", "bike")
    LABEL_BATCH = ("N", CLASSES, "H", "W")
    IMAGE_BATCH = ("N", 3, "H", "W")
    NET_RESULT = ("N", CLASSES, "H", "W")

    @shapecheck.check_args(img=(1, 3, 3, 4, 5), label_2=LABEL_BATCH)
    def f(img, label_1, label_2):
        ...

    v1 = torch.rand((1, 3, 3, 4, 5))
    v2 = torch.rand((1, 3, 4, 5))
    f(v1, v2, label_2=v2)


def test_readme_example():
    @shapecheck.check_args(x=(1, 3, 3, 4, 5), img=("N", ("R", "G", "B"), "H", "W"))
    def f(x, img):
        ...

    x = torch.rand((1, 3, 3, 4, 5))
    img = torch.rand((1, 3, 256, 256))
    f(x, img=img)
