import os
from pathlib import Path
from skimage import io
import tiledb
from typing import Optional
from tiledb.cloud.bioimg.ingestion import get_uris
import json
import typer
from pydantic import BaseModel, Field, validator
import concurrent.futures
from typing import List, Dict, Tuple
from tiledb import TileDBError


class PathValidator(BaseModel):
    path: str = Field(..., max_length=200)

    @validator("path")
    def validate_path(cls, path):
        if path.startswith("s3://"):
            # Path is from an S3 bucket, no validation needed
            return path

        if not os.path.isdir(path):
            raise ValueError("Invalid path. Please provide a valid directory path.")
        return path

class TileDBConverter:
    """
    This class converts images to TileDB arrays and performs ingestion into TileDB Cloud.

    Args:
        path: Input path where data lives
        output: Output directory where data will be saved
    """

    def __init__(self, path, output):
        self.path = path
        self.output = output
        self.words = ["s3", "polus-community-data"]
        self.token = None
        self.access_key = None
        self.secret_key = None
        self.region = None

    def contains_words(self) -> bool:
        """Check if given path contains strings given in list

        Returns:
            bool: True if word contains string in given list
        """
        basename = self.path.lower()
        return any(word.lower() in basename for word in self.words)

    def analyze_path(self) -> None:
        """Validate path and based on what strings path
        contains run correct methods
        """
        try:
            path_validator = PathValidator(path=self.path)
            path_validator.path  # Validate the path using the field assignment
            if self.contains_words():
                self.remote_conversion(self.output, self.path)
            else:
                self.local_conversion()
        except ValueError as e:
            typer.echo(str(e))

    def remote_conversion(self, output: str, path: str):
        """
        Perform remote conversion and ingestion into TileDB Cloud.

        Args:
            output: Output path where data will be saved.
            path: Input path where data is stored.
        """
        typer.echo("Running remote_conversion.")
        namespace = "Polus-Data"

        if not self.token:
            pass
            return

        samples, config_dict, vfs = self.configuration(output, namespace)
        self.ingest(samples, config_dict, vfs)
        self.collection_group(samples, namespace, output, path, vfs)

    def configuration(
        self, output: str, namespace: str
    ) -> Tuple[List[Tuple[str, str]], Dict, tiledb.VFS]:
        """
        Configure TileDB to read and write into an S3 bucket.

        Args:
            output: Output path where data will be saved.
            namespace: TileDB namespace where arrays will be registered.

        Returns:
            tuple: A tuple containing the samples, config_dict, and vfs.
        """

        config_dict = {
            "vfs.s3.aws_access_key_id": self.access_key,
            "vfs.s3.aws_secret_access_key": self.secret_key,
            "vfs.s3.region": self.region,
        }

        tiledb.cloud.login(token=self.token)

        cfg = tiledb.Config(params=config_dict)
        vfs = tiledb.VFS(cfg)
        path = [f"{self.path}/"]
        path_tdb = f"tiledb://{namespace}/{output}/image_arrays"
        samples = list(get_uris(path, path_tdb, cfg))

        return samples, config_dict, vfs

    def load_env_bash_file(self):
        """Load environment variables from the env.bash file."""
        try:
            with open("env.bash", "r") as file:
                env_lines = file.readlines()
                for line in env_lines:
                    if line.startswith("TILEDB_VFS_S3_AWS_ACCESS_KEY_ID"):
                        self.access_key = line.split("=")[1].strip()
                    elif line.startswith("TILEDB_VFS_S3_SECRET_ACCESS_KEY"):
                        self.secret_key = line.split("=")[1].strip()
                    elif line.startswith("TILEDB_VFS_S3_REGION"):
                        self.region = line.split("=")[1].strip()
                    elif line.startswith("TILEDB_API_TOKEN"):
                        self.token = line.split("=")[1].strip()
        except Exception as e:
            typer.echo(f"Error loading env.bash file: {str(e)}")

    def ingest(self, samples: List[tuple], config_dict: Dict, vfs: tiledb.VFS) -> None:
        """Use parallel processing and call process image method for
        each image in input path

        Args:
            samples: (input path where data lives, uri name of data)
            config_dict: Contains s3 configuration
            vfs: TileDB's virtual file system that reads and writes into s3
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for input_path, output_path in samples:
                futures.append(
                    executor.submit(
                        self.process_image, input_path, output_path, config_dict, vfs
                    )
                )
            concurrent.futures.wait(futures)

    def process_image(
        self, input_path: str, output_path: str, config_dict: Dict, vfs: tiledb.VFS
    ) -> None:
        """Reads in an image, validates it, and if the image is valid,
        converts it to a TileDB array from a NumPy array

        Args:
            input_path: Where data lives
            output_path: URI name of data
            config_dict: s3 configuration information
            vfs: TileDB's virtual file system that reads and writes into s3
        """
        try:
            with tiledb.scope_ctx(tiledb.cloud.Config()):
                with vfs.open(input_path) as src:
                    if self.is_valid_image(src):
                        image = io.imread(src)
                        tiledb.from_numpy(output_path, image)
                        typer.echo(f"Convert numpy array: {input_path} - Successful")
                    else:
                        typer.echo(f"Skipping invalid image: {input_path}")
        except Exception as e:
            typer.echo(f"Error processing image: {input_path}\n{str(e)}")

    def collection_group(
        self, samples: List[tuple], namespace: str, output: str, path: str, vfs: tiledb.VFS
    ) -> None:
        """Create folder structure inside TileDB and append metadata

        Args:
            samples: (input path where data lives, uri name of data)
            namespace: TileDB namespace where arrays will be registered
            output: output path where data will be saved
            vfs: TileDB's virtual file system that read and writes into s3

        Raises:
            e: Checks if folder being created already exists
        """
        basename = os.path.basename(path)
        with tiledb.scope_ctx(tiledb.cloud.Config()):
            tdb_uris = [x[1] for x in samples]
            # Create a collection
            collection_path = f"tiledb://{namespace}/{output}/{basename}"

            tiledb.Group.create(collection_path)
            typer.echo("Collection folder created.")
            with tiledb.Group(collection_path, "w") as collection:
                for member in tdb_uris:
                    try:
                        collection.add(member)
                    except TileDBError as e:
                        typer.echo(f"Skipping invalid member: {member}. Reason: {str(e)}")

    def local_conversion(self) -> None:
        """Call corresponding methods for local image conversion"""
        typer.echo("Ran local_conversion.")
        self.make_dir(self.output)
        self.convert_images(self.path, self.output)

    def make_dir(self, output: str) -> None:
        """Create folder structure on local disk

        Args:
            output: output path where data will be saved
        """
        Path(output).mkdir(parents=True, exist_ok=True)

    def convert_images(self, data_path: str, output: str) -> None:
        """Use parallel processing and call process local image method for
        each image in data path

        Args:
            data_path: Input path where data lives
            output: output path where data will be saved
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for file in os.listdir(data_path):
                image_path = f"{data_path}/{file}"
                futures.append(
                    executor.submit(self.process_local_image, image_path, output, file)
                )
            concurrent.futures.wait(futures)

    def process_local_image(self, image_path: str, output: str, file: str) -> None:
        """Reads in image and validates it, and if image is valid
        it will be converted to a TileDB array from numpy array

        Args:
            image_path: path of image
            output: output path where data will be saved
            file: name of image file
        """
        if self.is_valid_image(image_path):
            image = io.imread(image_path)
            output_path = f"{output}/{file}.tdb"
            tiledb.from_numpy(output_path, image)
            typer.echo(f"Convert numpy array: {image_path} - Successful")
        else:
            typer.echo(f"Skipping invalid image: {image_path}")

    def is_valid_image(self, image_path: str) -> bool:
        """Checks if image has dimensions

        Args:
            image_path: image file path

        Returns:
            bool: True if image dimensions are non-negative and non-zero
        """
        try:
            image = io.imread(image_path)
            image_width, image_height = image.shape[:2]
            if image_width > 0 and image_height > 0:
                return True
        except Exception:
            pass
        return False

    def load_s3_credentials(self):
        """Load S3 credentials from the env.bash file."""
        if not all([self.access_key, self.secret_key, self.region, self.token]):
            typer.echo("Missing S3 credentials in env.bash file.")
            return

        self.access_key = self.access_key.strip()
        self.secret_key = self.secret_key.strip()
        self.region = self.region.strip()
        self.token = self.token.strip()


app = typer.Typer()


@app.command()
def main(
    inp_dir: Optional[str] = typer.Option(None, "--inpDir", help="Input path where data lives"),
    out_dir: Optional[str] = typer.Option(None, "--outDir", help="Output directory where data will be saved")
):
    """Main function to convert images to TileDB arrays.

    Args:
        path: Input path where data lives.
        output: Output directory where data will be saved.
    """
    if inp_dir is not None and out_dir is not None:
        converter = TileDBConverter(inp_dir, out_dir)
        converter.analyze_path()

        if converter.contains_words():
            converter.load_env_bash_file()
            converter.load_s3_credentials()
            converter.remote_conversion(out_dir, inp_dir)
        else:
            converter.local_conversion()
    else:
        typer.echo("Please provide both path and output.")


if __name__ == "__main__":
    app()