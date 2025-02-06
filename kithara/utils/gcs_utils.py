"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from google.cloud.storage import Client, transfer_manager
import os
import time


def upload_file_to_gcs(
    local_file: str, gs_bucket_path: str, remove_local_file_after_upload=False
):
    """Uploads a single file to Google Cloud Storage.

    Args:
        local_file: Path to local file
        gs_bucket_path: GCS destination (e.g. "gs://my-bucket/path/file.txt" or "my-bucket/path/file.txt")
    """
    # Standardize bucket path format
    gs_bucket_path = gs_bucket_path.removeprefix("gs://")
    bucket_name = gs_bucket_path.split("/")[0]
    blob_name = gs_bucket_path[len(bucket_name) :].lstrip("/")

    print(f"-> Uploading {local_file} to {gs_bucket_path}...")
    # Upload file
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file)

    print(f"✅ Uploaded {local_file} to {bucket.name}/{blob_name}")

    if remove_local_file_after_upload:
        os.remove(local_file)
        print(f"✅ Deleted {local_file}")


def upload_folder_to_gcs(local_folder: str, gs_bucket_path: str, num_workers: int = 8):
    """Uploads all files from a local folder to Google Cloud Storage.

    Args:
        local_folder: Path to local folder (e.g. "data/images")
        gs_bucket_path: GCS destination (e.g. "gs://my-bucket/images" or "my-bucket/images")
        num_workers: Number of parallel upload workers
    """
    start_time = time.time()

    # Standardize bucket path format
    gs_bucket_path = gs_bucket_path.removeprefix("gs://")
    bucket_name = gs_bucket_path.split("/")[0]
    # Ensure destination ends with "/"
    destination_dir = gs_bucket_path[len(bucket_name) :]
    if destination_dir.startswith("/"):
        destination_dir = destination_dir[1:]
    if destination_dir != "" and not destination_dir.endswith("/"):
        destination_dir += "/"

    # Get files to upload
    files_in_local_folder = os.listdir(local_folder)
    # Set up GCS client
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    # Upload files in parallel
    results = transfer_manager.upload_many_from_filenames(
        bucket,
        files_in_local_folder,
        source_directory=local_folder,
        max_workers=num_workers,
        blob_name_prefix=destination_dir,
    )

    # Report results
    for name, result in zip(files_in_local_folder, results):
        if isinstance(result, Exception):
            print(f"Failed to upload {name}: {result}")
        else:
            print(f"✅ Uploaded {name} to {bucket.name}/{destination_dir}{name}")

    print(f"Upload completed in {time.time() - start_time}s")


def find_cache_root_dir():
    if "KERAS_HOME" in os.environ:
        cachdir = os.environ.get("KERAS_HOME")
    else:
        cachdir = os.path.expanduser(os.path.join("~", ".keras"))
    if not os.access(cachdir, os.W_OK):
        cachdir = os.path.join("/tmp", ".keras")
    return cachdir
