# TileDB Converter

The TileDB Converter is a Python command-line tool that allows you to convert images to TileDB arrays. It supports both local conversion and remote conversion using TileDB Cloud. 

## Prerequisites

Before running the TileDB Converter script, make sure you have the following dependencies installed:

- Python 3.x 
- `pip` package manager

## Installation

1. Clone the repository or download the Python script.
2. Navigate to the directory containing the script in your terminal.
3. Install the dependencies:
```<shell>

pip install -r requirements.txt

```

## Usage

To run the TileDB Converter script, execute the following command:
```<shell>

python tiledb_converter.py --inpDir <input_path> --outDir <output_directory> 

```

input_path (required): Path where the input data (images) is located. \
output_directory (required): Directory where the converted TileDB arrays will be saved. 


## Configuration

Before running the TileDB Converter, you need to configure the S3 credentials for remote conversion. Create an env.bash file in the root directory of the project and set the following environment variables:

```<shell>

TILEDB_VFS_S3_REGION=<s3_region>
TILEDB_VFS_S3_AWS_ACCESS_KEY_ID=<aws_access_key_id>
TILEDB_VFS_S3_SECRET_ACCESS_KEY=<aws_secret_access_key>
TILEDB_API_TOKEN=<api_token>

```

Replace <s3_region>, <aws_access_key_id>, <aws_secret_access_key> <api_token> with your S3 region, credentials, and token.

## Class and Methods

The main class in the TileDB Converter is the TileDBConverter class, which provides the following methods:

analyze_path(): Validates the input path and determines whether to perform local or remote conversion. \
remote_conversion(output): Performs remote conversion using TileDB Cloud. \
local_conversion(): Performs local conversion on the local file system. \
process_image(input_path, output_path, config_dict, vfs): Processes an image, converts it to a TileDB array, and saves it. \
collection_group(samples, namespace, output, vfs): Creates a folder structure inside TileDB and appends metadata. \
load_s3_credentials(): Loads the S3 credentials from the env.bash file. 

## Examples

### Local Conversion

To perform local conversion, provide the --inpDir and --outDir options:

```<shell>

python tiledb_converter.py --inpDir /path/to/input/images --outDir /path/to/output/directory

```

### Remote Conversion

To perform remote conversion using TileDB Cloud, provide the --inpDir, and --outDir options:

```<shell>

python tiledb_converter.py --inpDir s3://bucket/path/to/input/images --outDir s3://bucket/path/to/output/directory 

```