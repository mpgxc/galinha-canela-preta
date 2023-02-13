import cv2
import statistics
from glob import glob
import numpy as np
import tifffile
from sklearn.cluster import KMeans

from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage import (filters,
                     measure,
                     exposure,
                     morphology)
from tqdm import tqdm

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot(arr_images=[], grid=(2, 2), cmap="inferno"):

    fig = plt.figure(figsize=(20, 10))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=grid,
                     axes_pad=0.1)

    for ax, img in zip(grid, arr_images):
        ax.imshow(img, cmap)
        ax.axis('off')

    plt.show()
    

def binarize_image(arr):
    return arr > filters.threshold_triangle(arr)


def rescale_arr(arr, scale=255):
    return (arr * scale).astype('uint8')


def load_paths_images_sorted(path):
    """
    Carrega caminho da simagens e ordena em onrdem crescente os frames.
    """

    arr = []

    def parser_image_name(image_name):

        *_, name = image_name.split("/")
        name, *_ = name.split(".")

        return int(name)

    for index in tqdm(glob(f'{path}/*')):
        try:
            image_name = parser_image_name(index)

            arr.append(image_name)

        except:
            continue

    image_path_sorted = sorted(arr)

    def image_unique_name(x): return f"{path}/{x}.tif"

    return list(map(image_unique_name, image_path_sorted))


def load_images_from_paths(arr_paths, is_gray=False):
    arr_images = []

    if is_gray:
        for img_path in tqdm(arr_paths):

            try:
                frame = rgb2gray(tifffile.imread(img_path))

                is_valid_frame = statistics.mode(
                    binarize_image(frame).flatten())

                if not is_valid_frame:
                    continue

                arr_images.append(rescale_arr(frame))
            except:
                continue
    else:
        for img_path in tqdm(arr_paths):
            try:
                frame = tifffile.imread(img_path)

                is_valid_frame = statistics.mode(
                    binarize_image(frame).flatten())

                if not is_valid_frame:
                    continue

                arr_images.append(frame)
            except:
                continue
    return np.asarray(arr_images)


def auto_invert_image_mask(arr):
    """
    Calcula os pixels da imagem e inverte os pixels da imagem caso os pixels True > False
    Isso é uma forma de garatir que as mascaras tenham sempre o fundo preto = 0 e o ROI = 1
    """

    img = arr.copy()

    if statistics.mode(img.flatten()):
        img = np.invert(img)

    return img


def find_bighest_cluster_area(clusters):
    regions = measure.regionprops(clusters)

    all_areas = map(lambda item: item.area, regions)

    return max(all_areas)


def find_bighest_cluster(img):

    clusters = auto_invert_image_mask(img)

    clusters = measure.label(clusters, background=0)

    cluster_size = find_bighest_cluster_area(clusters)

    return morphology.remove_small_objects(clusters,
                                           min_size=(cluster_size - 1),
                                           connectivity=8)


def check_colision_border(mask):

    x, *_ = mask.shape

    left = mask[:1, ].flatten()
    right = mask[x - 1: x, ].flatten()
    top = mask[:, : 1].flatten()
    bottom = mask[:, x - 1: x].flatten()

    borders_flatten = [left, right, top, bottom]

    if np.concatenate(borders_flatten).sum() > 1:
        return True

    return False


def rule_of_three_percent_pixels(arr):

    def co_occurrence(arr):
        unique, counts = np.unique(arr, return_counts=True)

        return dict(zip(unique, counts))

    def ternary(value):
        return 0 if value is None else value

    def binarize_image(arr):
        return arr > filters.threshold_minimum(arr)

    image_coo = co_occurrence(arr)

    true_value = ternary(image_coo.get(True))
    false_value = ternary(image_coo.get(False))

    _100 = false_value + true_value

    return dict({
        'true_pixels': int((true_value * 100) / _100),
        'false_pixels': int((false_value * 100) / _100)
    })

def find_roi(img):

    binary_img = binarize_image(exposure.equalize_hist(img))

    best_cluster = find_bighest_cluster(binary_img)

    merged = binarize_image(binary_img + best_cluster)

    return binarize_image(find_bighest_cluster(merged))


def smoothing_mask_edges(mask):
    return binarize_image(filters.gaussian(mask, sigma=0.5))


def fill_smoothing_mask_edges(mask):
    mask = morphology.closing(mask, morphology.disk(9))

    mask = ndi.binary_fill_holes(mask)

    return mask

def apply_kmeans(img, k_clusters=3):

    img = np.array(img, dtype=np.float64) / 255

    img_reshaped = img.reshape((-1, 1))

    kmeans = KMeans(random_state=0, n_clusters=k_clusters).fit(img_reshaped)

    return kmeans.labels_.reshape(img.shape)


def find_best_larger_cluster(image_mask):

    clusters = image_mask.copy()

    if statistics.mode(clusters.flatten()):
        clusters = np.invert(clusters)

    clusters = measure.label(clusters, background=0)

    cluster_size = find_bighest_cluster_area(clusters)

    return morphology.remove_small_objects(
        clusters.astype(dtype=bool),
        min_size=(cluster_size-1),
        connectivity=8
    )


def read_image_frames_seq(path, ext='tif'):
    def parser_image_name(image_name):

        *_, name = image_name.split("/")
        name, *_ = name.split(".")

        return int(name)

    arr = []

    for index in glob(path + "/*"):
        try:
            image_name = parser_image_name(index)

            arr.append(image_name)

        except Exception:
            continue

    image_path_sorted = sorted(arr)

    def image_unique_name(x): return f"{path}/{x}.{ext}"

    return list(map(image_unique_name, image_path_sorted))


def build_volume_from_directory(arr_paths, is_gray=False):
    arr_images = []

    if is_gray:
        for img_path in arr_paths:

            frame = rgb2gray(tifffile.imread(img_path))

            if not statistics.mode(binarize_image(frame).flatten()):
                continue

            arr_images.append(frame)
    else:
        for img_path in arr_paths:

            frame = tifffile.imread(img_path)

            if not statistics.mode(binarize_image(frame).flatten()):
                continue

            arr_images.append(frame)

    return np.asarray(arr_images)


def find_broiler_roi(frame, background):
    def second_tecnique(frame):

        binary_frame = binarize_image(exposure.equalize_hist(frame))

        best_cluster = find_best_larger_cluster(binary_frame)

        merged = binarize_image(
            binary_frame.astype("uint8") + best_cluster.astype("uint8")
        )

        best_cluster = find_best_larger_cluster(merged)

        return binarize_image(best_cluster)

    def first_tecnique(frame, background):

        mask_bin = binarize_image(
            np.subtract(
                exposure.equalize_hist(background),
                exposure.equalize_hist(frame),
            )
        )

        best_cluster = find_best_larger_cluster(mask_bin)

        return binarize_image(best_cluster)

    mask_1 = second_tecnique(frame)
    mask_2 = first_tecnique(frame, background)
    mask = (mask_1 + mask_2).astype(dtype=bool)

    if statistics.mode(mask.flatten()):
        mask = np.invert(mask)

    # Arremata

    mask = find_best_larger_cluster(mask)
    mask = morphology.closing(mask, morphology.disk(5))
    mask = ndi.binary_fill_holes(mask)
    mask = filters.gaussian(mask, sigma=1.5)
    mask = binarize_image(mask)

    return mask.astype(dtype=bool)


def crop_image_box(image=None, shape=(100, 100), margin_pixel=30):

    x, y = shape

    return image[x - margin_pixel:
                 x + margin_pixel,
                 y - margin_pixel:
                 y + margin_pixel]


def find_center_mask(image_bin):

    props, *_ = measure.regionprops(
        measure.label(image_bin)
    )

    x, y = props.centroid

    return int(x), int(y)


def split_mask_v1(mask):

    thresh = mask.copy().astype(np.uint8)
    (contours, hierarchy) = cv2.findContours(thresh, 2, 1)

    i = 0

    for contour in contours:
        if cv2.contourArea(contour) > 20:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                continue
            points = []
            dd = []

            for i in range(defects.shape[0]):
                (s, e, f, d) = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                d = d / 256
                dd.append(d)

            for i in range(len(dd)):
                (s, e, f, d) = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                if dd[i] > 1.0 and dd[i] / np.max(dd) > 0.2:
                    points.append(f)

            i = i + 1
            if len(points) >= 2:
                for i in range(len(points)):
                    f1 = points[i]
                    p1 = tuple(contour[f1][0])
                    nearest = None
                    min_dist = np.inf
                    for j in range(len(points)):
                        if i != j:
                            f2 = points[j]
                            p2 = tuple(contour[f2][0])
                            dist = (p1[0] - p2[0]) * (p1[0] - p2[0]) \
                                + (p1[1] - p2[1]) * (p1[1] - p2[1])
                            if dist < min_dist:
                                min_dist = dist
                                nearest = p2

                    cv2.line(thresh, p1, nearest, [0, 0, 0], 2)
    return thresh