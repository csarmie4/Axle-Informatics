from urllib.request import urlretrieve
from zipfile import ZipFile
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import logging
from typing import Optional
import time
import re
import shutil

from bfio import BioWriter
from skimage import io
import numpy as np
import cv2
from tqdm import tqdm

DATASET = "BBBC020"
global num_workers
num_workers = max(cpu_count(), 2)
logger = logging.getLogger(f"{DATASET}")


def save(image: np.ndarray, out_file: Path) -> None:
    """Saving images in ometif or omezarr format

    Args:
        image: Input image array
        out_file: Path to save ometif or omezarr image
    """
    num_channels = 1 if len(image.shape) == 2 else image.shape[3]

    with BioWriter(out_file) as bw:
        bw.X, bw.Y, bw.Z, bw.C = (image.shape[1], image.shape[0], 1, num_channels)
        bw.dtype = image.dtype
        bw[:] = image

    return


def makedirectory(
    root: Optional[Path] = Path(".data"), sub_path: Optional[str] = None
) -> None:
    sub_path = "" if sub_path == None else sub_path

    if not root.joinpath(sub_path).exists():
        root.joinpath(sub_path).mkdir(parents=True, exist_ok=True)

    return


def download_files(root: Optional[Path] = Path(".data")) -> None:
    """Downloading zipped files of images and ground truth images

    Args:
        root: Root of the path where downloaded data will be saved
    """

    raw_path = root.joinpath("raw")
    urls = [
        "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_outlines_nuclei.zip",
        "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_outlines_cells.zip",
    ]

    def download_url(url):
        zip_path, _ = urlretrieve(url)

        with ZipFile(zip_path, "r") as zfile:
            zfile.extractall(raw_path)

        return

    num_workers = max(cpu_count(), 2)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        threads = []

        for url in urls:
            threads.append(executor.submit(download_url, url=url))

        for f in tqdm(
            as_completed(threads), desc=f"Downloading data", total=len(threads)
        ):
            f.result()

    images_path = raw_path.joinpath("BBBC020_v1_images")

    for folder in images_path.iterdir():
        src = images_path.joinpath(folder.name)
        dst = raw_path.joinpath(folder.name)

        if raw_path.joinpath(folder.name).exists():
            continue

        shutil.move(str(src), str(dst))

    shutil.rmtree(str(images_path))

    return


def process_images(
    root: Optional[Path] = Path(".data"), file_extension: Optional[str] = ".ome.zarr"
) -> None:
    """Standardizes raw images

    Args:
        root: Root of the path where standard images will be saved
        file_extension: Formats to save images ('.ome.tif', '.ome.zarr'). By default images are saved in '.ome.zarr' format
    """

    makedirectory(root, "standard/intensity")

    treatment = {"Kontrolle": 0, "15min": 1, "30min": 1, "1h": 1, "2h": 1, "24h": 1}

    timepoint = {"Kontrolle": 0, "15min": 1, "30min": 2, "1h": 3, "2h": 4, "24h": 5}

    channel = {"c1": 0, "c5": 1}

    standard_path = root.joinpath("standard")
    raw_path = root.joinpath("raw")

    pattern = "jw-(Kontrolle)([0-9]+)_(\w+).TIF|jw-(\w+) ([0-9]+)_(\w+).TIF"

    threads = []
    num_workers = max(cpu_count(), 2)

    # Iterate over each directory in raw but ignore BBC020_v1_outlines_cells and BBC020_v1_outlines_nuclei
    for directory in [obj for obj in raw_path.iterdir() if obj.is_dir()]:
        if (
            directory.name == "BBC020_v1_outlines_cells"
            or directory.name == "BBC020_v1_outlines_nuclei"
        ):
            continue

        # In each directory, iterate over each image
        for file in directory.iterdir():
            m = re.match(pattern, file.name)

            # Skip over the multichannel image
            if m == None:
                continue

            name_info = [elem for elem in m.groups() if elem != None]

            out_file = f"p0{treatment[name_info[0]]}_t0{timepoint[name_info[0]]}_r0{name_info[1]}_c0{channel[name_info[2]]}{file_extension}"
            out_file = standard_path.joinpath("intensity", out_file)

            image = io.imread(directory.joinpath(file.name))

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                if name_info[2] == "c1":
                    threads.append(executor.submit(save, image[:, :, 0], out_file))
                else:
                    threads.append(executor.submit(save, image[:, :, 2], out_file))

    for f in tqdm(as_completed(threads), desc=f"Saving images", total=len(threads)):
        f.result()

    return


def process_masks(
    root: Optional[Path] = Path(".data"), file_extension: Optional[str] = ".ome.zarr"
) -> None:
    """Combines the individual cell masks into images with multiple channels and saves the images

    Args:
        root: Root of the path where the multi-channel mask images will be saved
        file_extension: Formats to save images ('.ome.tif', '.ome.zarr'). By default images are saved in '.ome.zarr' format
    """

    def overlaps(image: np.ndarray, indices) -> bool:
        """Checks to see if any points in indices correspond to nonzero pixels in image. If they
        do, then this indicates overlap

        Returns:
            True if there s overlap. False otherwise
        """

        for i, j in zip(indices[0], indices[1]):
            if image[i, j] != 0:
                return True

        return False

    def combine_and_label(path: Path) -> dict:
        """Combines the individual cell masks and stores them in numpy ndarrays. The ndarrays are the
        different channels of each image. Each cell gets a unique label as it is added to an ndarrray

        Args:
            path: A path to a directory that contains the masks
        """
        # final_images has the following structure {group : [channels, label]}. group is the common prefix
        # for a group of masks. channels is a list of channels (numpy ndarrays) for the final image. label
        # is the current label associated with the final image.
        final_images = {}

        for dir in [obj for obj in images_path.iterdir() if obj.is_dir()]:
            if (
                dir.name == "BBC020_v1_outlines_cells"
                or dir.name == "BBC020_v1_outlines_nuclei"
            ):
                continue

            # Data for jw-30min is missing so we ignore it
            if re.search("jw-30min", dir.name) == None:
                final_images[dir.name] = [
                    [np.zeros((1040, 1388, 1), dtype=np.uint8)],
                    1,
                ]
            else:
                logger.info(
                    f"Images associated with the 30 min timepoint are missing outline data"
                )

        # Iterate over all outline images
        for im in tqdm(sorted(path.iterdir()), desc="Processing masks"):
            pattern = "(jw-Kontrolle[0-9]+)_\w+_(\d+).TIF|(jw-\w+ [0-9]+)_\w+_(\d+).TIF"
            m = re.match(pattern, im.name)
            file_name_info = [obj for obj in m.groups() if obj != None]

            # The group name of the mask image
            group = file_name_info[0]

            try:
                mask_image = io.imread(path.joinpath(im.name))
            except Exception as e:
                print("Image name: " + im.name + ", Error: " + str(e))
                continue

            channels = final_images[group][0]
            label = final_images[group][1]

            # Change the cell in the mask image from all white to whatever the cell's label is
            mask_image[mask_image == 255] = label

            indices = mask_image.nonzero()

            # The channel that we want to add the cell to. Initially set to the first channel
            target_channel = channels[0]

            # If the new cell overlaps a cell in target_channel, try to add it to the next channel
            if overlaps(target_channel, indices):
                for channel in channels[1:]:
                    if not overlaps(channel, indices):
                        for i, j in zip(indices[0], indices[1]):
                            channel[i, j] = mask_image[i, j]
                        break
                else:
                    # If the cell overlaps another cell in every channel, then a new channel is added
                    # for that cell
                    channels.append(np.zeros((1040, 1388, 1), dtype=np.uint8))
                    target_channel = channels[-1]

                    for i, j in zip(indices[0], indices[1]):
                        target_channel[i, j] = mask_image[i, j]
            # Otherwise, add the cell to the first channel
            else:
                for i, j in zip(indices[0], indices[1]):
                    target_channel[i, j] = mask_image[i, j]

            # Increment label
            final_images[group][1] += 1

        return final_images

    def standardize(final_images: dict, chan: str) -> None:
        """Standardize the multi-channel mask images

        Args:
            final_images: Maps mask name to final combined, labeled mask
            chan: Indicates whether the image is of the nuclear or cell channel
        """

        treatment = {"Kontrolle": 0, "15min": 1, "30min": 1, "1h": 1, "2h": 1, "24h": 1}

        timepoint = {"Kontrolle": 0, "15min": 1, "30min": 2, "1h": 3, "2h": 4, "24h": 5}

        channel = {"cell": 0, "nuclei": 1}

        for group, channels in tqdm(final_images.items(), desc="Saving combined masks"):
            pattern = "jw-(Kontrolle)([0-9]+)|jw-(\w+) ([0-9]+)"
            m = re.match(pattern, group)
            group_info = [obj for obj in m.groups() if obj != None]
            out_file = f"p0{treatment[group_info[0]]}_t0{timepoint[group_info[0]]}_r0{group_info[1]}_c0{channel[chan]}{file_extension}"

            image = cv2.merge(
                channels[0], np.zeros(channels[0][0].shape, dtype=np.uint8)
            )
            image = np.expand_dims(image, axis=2)
            save(image, Path(root, f"standard/masks/{out_file}"))

        return

    makedirectory(root, "standard/masks")

    cell_outline_path = root.joinpath("raw/BBC020_v1_outlines_cells")
    nuclei_outline_path = root.joinpath("raw/BBC020_v1_outlines_nuclei")
    images_path = root.joinpath("raw")

    final_cells = combine_and_label(cell_outline_path)
    final_nuclei = combine_and_label(nuclei_outline_path)

    standardize(final_cells, "cell")
    standardize(final_nuclei, "nuclei")

    return


def BBBC020_data(
    root: Optional[Path] = Path(".data"), file_extension: Optional[str] = ".ome.zarr"
) -> None:
    """Downloads and standardizes the BBBC020 dataset.

    Args:
        root: Root of the path where all data will be saved
        file_extension: Formats to save images ('.ome.tif', '.ome.zarr'). By default images are saved in '.ome.zarr' format
    """
    start = time.perf_counter()

    logger.info(f"Making raw and standard directories")
    makedirectory(root, "raw")
    makedirectory(root, "standard")
    logger.info(f"Downloading Zipped files")
    download_files(root)
    logger.info(f"Standardizing images")
    process_images(root, file_extension)
    logger.info(f"Processing masks")
    process_masks(root, file_extension)

    end = time.perf_counter()
    time_taken = round(end - start, 2)

    logger.info(f"Total time for processing {DATASET} is {time_taken}")

    return


def main():
    if len(sys.argv) == 1:
        print(f"Please define the function to be executed")
        print("Usage: python3 standard.py BBBC020_data <root> <file_extension>")
        sys.exit()
    else:
        function = sys.argv[1]
        if function == "BBBC020_data":
            root = str(sys.argv[2])
            root = Path(root).absolute()
            file_extension = str(sys.argv[3])
            BBBC020_data(root, file_extension)


if __name__ == "__main__":
    main()
