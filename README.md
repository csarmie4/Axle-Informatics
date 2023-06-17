# Axle-Informatics
This repo includes work I have done with Axle Informatics | NIH-NCATS 

## What can be found
### TileDB-testing
● Designed and implemented a Python class for converting image data to TileDB arrays. The script supports
local and remote conversion, intelligently detects the source location and utilizing parallel processing which
significantly improved the processing speed by utilizing the available CPU cores efficiently. \
● Integrated the script with Amazon S3 for remote conversions, allowing seamless ingestion of data from S3
buckets. Implemented authentication and utilized the TileDB API to configure the S3 connection and
perform read and write operations. Tested on both EC2 instances and local machines. \
● Implemented validation checks to ensure the images are valid before conversion, considering image
dimensions. Handled potential errors and exceptions during the conversion process, providing informative
messages for troubleshooting.

### BBBC020
Dataset used can be found here : https://bbbc.broadinstitute.org/BBBC020 \
● Facilitated a data processing pipeline in Python using libraries such as bfio, skimage, and NumPy. The
pipeline downloads zipped files of images, renames files to follow a file naming convention, combines
individual cell masks into multi-channel images, and saves the processed data in a specified format.