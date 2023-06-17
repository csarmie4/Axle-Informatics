# BBBC020 Data
This data comes from the [BBBC020](https://bbbc.broadinstitute.org/BBBC020) dataset. The image set contains 25 samples, each with three images, totaling 75 images. Each sample was stained with DAPI and CD11b/APC. There are two separate images for the stained sample and one merged image. The ground truth includes outlines of each cell for each sample. There are outlines of nuclei and cell bodies. These outlines were added to help evaluate segmentation performance for overlapping cells. Cell segmentation for these images is challeneging because the cells are irregularly shaped and they overlap eachother. For this reason, this dataset can be used to assess cell segmentation algorithms to see how well thay can do with overlapping cell data. All images are 1388x1040 pixels.

## Sample Images
<img src="https://user-images.githubusercontent.com/79129803/232678950-44177ef8-a7f9-497c-abd1-0fee29238bed.jpg" width="250" height="250" alt="CD11b/APC" title="CD11b/APC"> <img src="https://user-images.githubusercontent.com/79129803/232679010-9d98f69d-7ef4-451d-b483-a67831386b1c.jpg" width="250" height="250" alt="DAPI" title="DAPI"> <img src="https://user-images.githubusercontent.com/79129803/232678686-bd9d0ca3-ace2-485b-b212-3f59b3a93940.jpg" width="250" height="250" alt="Merged" title="Merged">

## File Naming Convention
The naming format for the standardized data is `p{p+}_t{t+}_r{r+}_c{c+}.\<extension\>` where extension is `ome.tif` or `ome.zarr`.

**p** = Control or treated contition (0-1)

**t** = Timepoint for the treatment (0-5)

**r** = Replicate number

**c** = Cell or nuclei (0-1)

### Condition Index Reference
| Index | Label |
| ----------- | ----------- |
| Control | 0 |
| Treated | 1 |

### Timepoint Index Reference
| Index | Label |
| ----------- | ----------- |
| Control | 0 |
| 15 min | 1 |
| 30 min | 2 |
| 1 hr | 3 |
| 2 hr | 4 |
| 24 hr | 5 |

### Channel Index Reference
| Index | Label |
| ----------- | ----------- |
| Cell | 0 |
| Nuclei | 1 |

## Dataset Size
The raw data has a total size of ~2.08 GB. The nuclei outlines have a total size of ~1.01 GB, the cell outlines have a total size of ~747.2 MB, and the image set has a total size of ~325.1 MB.

The size of the standard data depends on the file extension. The standard data has a total size of ~100 MB when using `.ome.tif` and ~45 MB when using `.ome.zarr`.

## Contents
### Raw Data
`./raw` - A directory for each sample containing the raw data for that sample and two directories for cell and nuclei outlines.

`./raw/BBC020_v1_outlines_cells` - The cell outlines

`./raw/BBC020_v1_outlines_nuclei` - The nuclei outlines

### Standard Data
`./standard/intensity` - The standard images

`./standard/masks` - The outlines associated with each image combined into a single, labeled, multi-channel, standardized image

#### How the Outlines are Standardized
Each sample has nucleus and cell outlines associated with it. The outlines are binary masks of individual cells or nuclei. These outlines are combined into a single, labeled, multi-channel, standardized image and stored in the `./standard/masks` directory. Each individual cell gets its own unique label in the combined image. If there is overlapp when a new cell is being added to the final image, then that cell is moved to a different channel. There are no overlapping cells in a single channel. The final image is saved in `.ome.tif` or `.ome.zarr` format and follows the standard naming convention described above.