#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
simple client script
"""

import base64
import io

from PIL import Image
import requests


def main():
    """
    main
    """

    with open("33.jpg", "rb") as inputfile:
        data = inputfile.read()

    post_data = {"image": base64.b64encode(data).decode("utf-8"),
                 "get_img_flg": True, "thresh": 0.5}
    res = requests.post("http://192.168.1.74:8022/detect", json=post_data).json()

    if "pred_img" in res:
        pred_data = base64.b64decode(res["pred_img"])
        img = Image.open(io.BytesIO(pred_data))
        img.save("test.png")
        del res["pred_img"]
        print(res)

if __name__ == "__main__":
    main()
