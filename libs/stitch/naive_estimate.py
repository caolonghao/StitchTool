from scipy.ndimage import gaussian_filter
from skimage import io
import numpy as np


def im2col(img, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2  # 默认stride为1

    H, W = img.shape
    out_h = (H + 2 * pad - kernel_size) // stride + 1
    out_w = (W + 2 * pad - kernel_size) // stride + 1

    img = np.pad(
        img, [(pad, pad), (pad, pad)], "edge"
    )  # padding部分的mean、std是有影响的，是否需要单独拉出来算？
    col = np.zeros((kernel_size, kernel_size, out_h, out_w))

    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            col[y, x, :, :] = img[y:y_max:stride, x:x_max:stride]
    col = col.transpose(2, 3, 0, 1).reshape(out_h * out_w, -1)
    return col


class NaiveEstimate:
    # windows size is recommended to be an odd number
    def __init__(self, debug=False, window_size=5, select_scale_factor=0.5) -> None:
        self.debug = debug
        self.window_size = window_size
        self.select_scale_factor = select_scale_factor

    def image_deviation_select(self, images, scale_factor=0.5):
        reshaped_images = images.reshape(images.shape[0], -1)
        std_deviation = np.std(reshaped_images, axis=1)
        threshold = std_deviation.min() * (1.0 + scale_factor)
        valid_pos = std_deviation <= threshold

        return images[valid_pos]

    def __call__(self, img_stack, gaussian_sigma=0.6):
        # gaussian filter
        if gaussian_sigma > 0:
            img_stack = gaussian_filter(img_stack, sigma=gaussian_sigma)
            print("gaussian filter sigma={:.1f} done".format(gaussian_sigma))

        # remove added all-zero image
        # new_img_stack = []
        # for index in range(img_stack.shape[0]):
        #     img = img_stack[index]
        #     if np.max(img) != 0:
        #         new_img_stack.append(img)
        # img_stack = np.array(new_img_stack)

        # pixel-wise sort, reverse order
        reconstructed_imgs = np.sort(img_stack.T, axis=2).T[::-1].astype(np.float64)

        # debug 展示
        if self.debug:
            for i, img in enumerate(reconstructed_imgs):
                io.imsave(f"distortion_{i}.tif", img)

        distortions, bgs = [], []

        img_nums = reconstructed_imgs.shape[0]
        # 检查图片数量，如果图片数量过少，直接取最小值作为背景，因为此时对于背景的估计不可靠，会引入伪影
        if img_nums < 20:
            bg_value = np.min(reconstructed_imgs)
            bg = np.ones_like(reconstructed_imgs[0]) * bg_value
        else:
            # TODO:后续可以考虑在平滑度和亮度间做一个更好的决策,比如加权得分排序之类的；或者在最平滑的中选最暗的？
            dark_img_nums = int(reconstructed_imgs.shape[0] * 0.1)
            valid_dark_imgs = self.image_deviation_select(
                reconstructed_imgs[-dark_img_nums:], self.select_scale_factor
            )
            for i, img in enumerate(valid_dark_imgs):
                bgs.append([img, self.LCoV_evaluate(img), i])

            bgs.sort(key=lambda x: x[1])
            LCoV_threshold = bgs[0][1] * (1.0 + 0.005)

            max_percentage = 0.0
            min_percentage = 1.0
            valid_bgs = []
            for img, LCoV_score, index in bgs:
                if LCoV_score < LCoV_threshold:
                    valid_bgs.append(img)
                else:
                    break

            bg = sum(valid_bgs) / len(valid_bgs)

        # TODO: evaluate 相对较慢，可以考虑减少选取的图片量
        # 现在通过实验发现大部分最优点都在30%之前，只有极少数的会出现在前50%
        max_bright_img_nums = int(reconstructed_imgs.shape[0] * 0.8)
        min_bright_img_nums = int(reconstructed_imgs.shape[0] * 0.1)
        valid_bright_imgs = self.image_deviation_select(
            reconstructed_imgs[min_bright_img_nums:max_bright_img_nums] - bg,
            self.select_scale_factor,
        )
        for i, img in enumerate(valid_bright_imgs):
            distortions.append([img, self.LCoV_evaluate(img), i])

        # 平滑度阈值不超过最低的 0.5%
        distortions.sort(key=lambda x: x[1])
        LCoV_threshold = distortions[0][1] * (1.0 + 0.005)

        max_percentage = 0.0
        min_percentage = 1.0
        valid_flat = []
        for img, LCoV_score, index in distortions:
            if LCoV_score < LCoV_threshold:
                valid_flat.append(img)
                # max_percentage = max(max_percentage, index / len(reconstructed_imgs))
                # min_percentage = min(min_percentage, index / len(reconstructed_imgs))
                # print(
                #     "Original image index percentage:{:.2f}".format(
                #         index / len(reconstructed_imgs)
                #     )
                # )
            else:
                # print("max percentage: {:.2f}".format(max_percentage))
                # print("min percentage: {:.2f}".format(min_percentage))
                break

        flat = sum(valid_flat) / len(valid_flat)

        # print("valid flat numbers:", len(valid_flat))
        # io.imsave("flat.tif", distortions[0][0].astype(np.uint16))
        # TODO:数学推导模拟

        # past-gaussian-filter 影响不是很大，省略也可
        flat = gaussian_filter(flat, sigma=10, mode="nearest")
        bg = gaussian_filter(bg, sigma=10, mode="nearest")

        return flat, bg

    def LCoV_evaluate(self, distortion_img):
        cols = im2col(distortion_img, kernel_size=self.window_size, stride=1)
        LCoV_sum = np.sum(np.std(cols, axis=1) / np.mean(cols, axis=1))

        return LCoV_sum
