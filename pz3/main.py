import numpy as np
import cv2
import matplotlib.pyplot as plt

for i in range(1, 4, 1):
    image = cv2.imread(f'img_{i}.jpeg')
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, l, s = cv2.split(image_hls)
    shape = l.shape
    l[l < 230] = 0
    mask = np.zeros(shape, dtype=np.uint8)
    mask[50:300, :350] = 255
    l_and_mask = cv2.bitwise_and(mask, l)
    sum_y = l_and_mask.sum(axis=0)
    max_sum_x = max(sum_y)
    x = [i for i in range(len(sum_y))]
    max_vals_x = [i for i, val in enumerate(sum_y) if val == max_sum_x]
    avg_max_vals_x = np.sum(max_vals_x)/len(max_vals_x)

    sum_x = l_and_mask.sum(axis=1)
    max_sum_y = max(sum_x)
    y = [i for i in range(len(sum_x))]
    max_vals_y = [i for i, val in enumerate(sum_x) if val == max_sum_y]
    avg_max_vals_y = np.sum(max_vals_y) / len(max_vals_y)

    p_start_x = int(avg_max_vals_x - 10)
    if p_start_x < 0:
        p_start_x = 0

    p_end_x = int(avg_max_vals_x + 10)
    if p_end_x > l_and_mask.shape[1]:
        p_end_x = int(l_and_mask.shape[1])

    p_start_y = int(avg_max_vals_y - 10)
    if p_start_y < 0:
        p_start_y = 0

    p_end_y = int(avg_max_vals_y + 10)
    if p_end_y > l_and_mask.shape[0]:
        p_end_y = int(l_and_mask.shape[0])

    mask[:, :] = 0
    mask[p_start_y:p_end_y, p_start_x:p_end_x] = 255
    l_and_mask = cv2.bitwise_and(mask, l)

    ret, thresh = cv2.threshold(l_and_mask, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    M = cv2.moments(thresh)
    c_x = M["m10"] / M["m00"]
    c_y = M["m01"] / M["m00"]


    plt.figure(figsize=(8, 8))
    cv2.line(image_rgb, (int(c_x), 0), (int(c_x), image_rgb.shape[0]), (255, 0, 0), 1)
    cv2.line(image_rgb, (0, int(c_y)), (image_rgb.shape[1], int(c_y)), (255, 0, 0), 1)
    plt.imshow(image_rgb)
    plt.show()
