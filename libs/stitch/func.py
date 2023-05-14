# coding: utf-8
from os import path

import numpy as np
from scipy import ndimage
from scipy.ndimage import median_filter

from skimage import io

from pystackreg import StackReg
from basicpy import BaSiC
import jax
import cv2

jax.config.update("jax_platform_name", "cpu")

import time
from .flat_estimate import FlatEstimate
from .naive_estimate import NaiveEstimate
from .bagging import Bagging
from .utils import *


class StitchTool:
    def __init__(self, flat=None, bg=None) -> None:
        self.flat = flat
        self.bg = bg
        self.sr = StackReg(StackReg.RIGID_BODY)
        self.flat_estimate = FlatEstimate()
        self.naive_estimate = NaiveEstimate()

    def set_flat(self, flat_info):
        self.flat_info = flat_info

    def correct(self, src, bias=0):
        if "estimate" in self.flat_info:
            flat, bg = self.flat_estimate(src, bias)
        elif "NaiveEstimate" in self.flat_info:
            src = cut_light(src, min_num=0.5, max_num=95)
            src = grid_noise_filter(src)
            # src = cross_signal_filter(src, size=5)
            start_time = time.time()
            flat, bg = self.naive_estimate(src, bias)
        elif "BaSic" in self.flat_info:
            src = cut_light(src, max_num=95)
            src = grid_noise_filter(src)
            start_time = time.time()
            basic = BaSiC(get_darkfield=True, smoothness_flatfield=1)
            basic.fit(src)
            flat, bg = basic.flatfield, basic.darkfield
            # flat, bg = basicpy.basic(src, darkfield=True)

        elif self.flat_info["flat"] is None:
            # src = cut_light(src)
            return src
        else:
            flat = io.imread(self.flat_info["flat"])
            bg = io.imread(self.flat_info["bg"])

        end_time = time.time()
        save_folder = r"D:\MyCode\Light Field Correction\DisplayData\correction"
        cv2.imwritemulti(path.join(save_folder, "origin.tif"), src)

        src = reverse_with_flat_bg(src, flat, bg)
        src = src.astype(np.uint16)

        name = "BaSiC" if "BaSic" in self.flat_info else "NaiveEstimate"
        flat_name = "BaSiC" if "BaSic" in self.flat_info else "Naive"
        # if name == "NaiveEstimate":
        cv2.imwrite(path.join(save_folder, flat_name + "flat.tif"), flat)
        cv2.imwrite(path.join(save_folder, name + "bg.tif"), bg)
        cv2.imwritemulti(path.join(save_folder, name + "corrected.tif"), src)

        print("---- Correct time: {:.2f}s".format(end_time - start_time))
        return src

    def average_img(self, images, average_num, is_registration=False):
        img_avg = np.zeros(
            (images.shape[0] // average_num, images.shape[1], images.shape[2]),
            dtype=np.uint16,
        )
        for i in range(images.shape[0] // average_num):
            img_stack = images[i * average_num : (i + 1) * average_num]
            if is_registration:
                out_mean_af = self.sr.register_stack(
                    img_stack, axis=0, reference="previous"
                )  # 输出的是形变矩阵
                img_stack = self.sr.transform_stack(img_stack)
            img_avg[i] = np.mean(img_stack, axis=0)
        return img_avg

    def stitch(
        self,
        img_path,
        save_path,
        grid_shape,
        average_num=1,
        bias=-1500,
        overlay=0.1,
        fill=True,
        is_registration=False,
    ):
        if is_registration:
            print("开启配准,耗时很长请等待...")
        [row_num, col_num] = grid_shape
        images = io.imread(img_path).astype(np.float32)
        if images.shape[0] != grid_shape[0] * grid_shape[1] * average_num:
            new_img = np.zeros(
                (
                    grid_shape[0] * grid_shape[1] * average_num,
                    images.shape[1],
                    images.shape[2],
                ),
                dtype=np.uint16,
            )
            if fill:
                print("图片缺帧,已在尾部补帧")
                new_img[: images.shape[0]] = images
            else:
                print("图片缺帧,已在头部补帧")
                new_img[-images.shape[0] :] = images

            images = new_img
        images = self.average_img(images, average_num, is_registration)
        images = self.correct(images, bias)

        output = merge_images(images, row_num, col_num)
        # 后处理中值滤波，能够去串扰
        output = median_filter(output, size=3)
        print(">>>>  finish", save_path)
        io.imsave(save_path, output)
        return output


if __name__ == "__main__":
    aaa = np.zeros((10, 512, 512))
    aaa = reg_img_stack(aaa)
    print(aaa.shape)
    # stitch_tool = StitchTool()
    # stitch_tool.stitch(img_path='CellVideo.tif', save_path='CellVideo_stitch.tif', grid_shape=[20,22])
