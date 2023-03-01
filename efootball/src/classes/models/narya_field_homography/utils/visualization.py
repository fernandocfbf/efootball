import matplotlib.pyplot as plt
import numpy as np

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()

def rgb_template_to_coord_conv_template(rgb_template):
    assert isinstance(rgb_template, np.ndarray)
    assert rgb_template.min() >= 0.0
    assert rgb_template.max() <= 1.0
    rgb_template = np.mean(rgb_template, 2)
    x_coord, y_coord = np.meshgrid(
        np.linspace(0, 1, num=rgb_template.shape[1]),
        np.linspace(0, 1, num=rgb_template.shape[0]),
    )
    coord_conv_template = np.stack((rgb_template, x_coord, y_coord), axis=2)
    return coord_conv_template

def merge_template(img, warped_template):
    valid_index = warped_template[:, :, 0] > 0.0
    overlay = (
        img[valid_index].astype("float32")
        + warped_template[valid_index].astype("float32")
    ) / 2
    new_image = np.copy(img)
    new_image[valid_index] = overlay
    return new_image