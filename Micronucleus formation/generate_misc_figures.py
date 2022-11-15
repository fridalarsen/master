import numpy as np
import matplotlib.pyplot as plt

import skimage.io as skio
import skimage.filters as skfilt

from program_v1 import pre_process, segmentation_v1

# figure for comparison of nucleus size 05MAY2022 and 15MAR2022
image1 = skio.imread("./MN_raw_data/05MAY2022/3.90gy-p1-1/1_ch01.tif")
image2 = skio.imread("./MN_raw_data/15MAR2022/4.06gy-p5-3/2_ch01.tif")

figure, axes = plt.subplots(1,2)

axes[0].imshow(image1, cmap="Reds_r")
axes[0].set_title("05MAY2022", fontsize=15)
axes[0].axis("off")

axes[1].imshow(image2, cmap="Reds_r")
axes[1].set_title("15MAR2022", fontsize=15)
axes[1].axis("off")

figure.suptitle("Images from nucleus channel", fontsize=20)
plt.tight_layout()
plt.savefig("./Figures/compare_nuc_size.png", bbox_inches="tight", dpi=200)
plt.show()

# figure showing histogram equalization on image with background light
# gradient
image = skio.imread("./MN_raw_data/06SEP2022/0gy-pNA-1/2_ch00.tif")
filtered_image = pre_process(image, filter="Blur", size=10, equalize="Regular")

thresh = skfilt.threshold_otsu(filtered_image)

figure = plt.figure(constrained_layout=True, figsize=(6,6))
subfigs = figure.subfigures(2, 1, wspace=0.1)

top = subfigs[0].subplots(1,2)
top[0].imshow(image, cmap="Greens_r")
top[0].set_title("Original image", fontsize=20)
top[0].axis("off")
top[1].imshow(filtered_image, cmap="Reds_r")
top[1].set_title("Filtered image", fontsize=20)
top[1].axis("off")

bottom = subfigs[1].subplots(1,1)
bottom.imshow(filtered_image > thresh, cmap="Purples")
bottom.set_title("Otsu mask of filtered image", fontsize=20)
bottom.axis("off")

plt.savefig("./Figures/light_gradient_example1.png", bbox_inches="tight", dpi=200)
plt.show()

multi_otsu_thresh = skfilt.threshold_multiotsu(filtered_image)
figure, axes = plt.subplots(1,2, figsize=(10,5))
axes[0].imshow(filtered_image, cmap="Reds_r")
axes[0].set_title("Filtered image", fontsize=20)
axes[0].axis("off")

axes[1].hist(filtered_image.ravel(), bins=200)
ylim = axes[1].get_ylim()
axes[1].vlines(thresh, ylim[0], ylim[1], label="Otsu\nthreshold", color="orange", linewidth=3)
axes[1].vlines(multi_otsu_thresh[-1], ylim[0], ylim[1], label="Multi-Otsu\nthreshold", color="hotpink", linewidth=3)
axes[1].set_ylim(ylim)
axes[1].set_title("Histogram of image", fontsize=20)
axes[1].set_xlabel("Intensity", fontsize=15)
axes[1].set_ylabel("Number of pixels", fontsize=15)
axes[1].legend(fontsize=13, loc="upper right")

figure.tight_layout()
plt.savefig("./Figures/light_gradient_example2.png", bbox_inches="tight", dpi=200)
plt.show()
