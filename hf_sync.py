"""
Hugging Face dataset sync utility.

Usage:
    python hf_sync.py upload <local_path> <repo_path>
    python hf_sync.py upload-folder <local_folder> <repo_folder>
    python hf_sync.py download <repo_path> <local_path>
    python hf_sync.py download-folder <repo_folder> <local_folder>
    python hf_sync.py delete <repo_path>
    python hf_sync.py delete-pattern <glob_pattern>

Examples:
    python hf_sync.py upload data/embeddings/s2_train_glove_embeddings.npy data/embeddings/s2_train_glove_embeddings.npy
    python hf_sync.py upload-folder data/embeddings data/embeddings
    python hf_sync.py download data/embeddings/s2_train_glove_embeddings.npy data/embeddings/s2_train_glove_embeddings.npy
    python hf_sync.py download-folder data/embeddings data
    python hf_sync.py delete data/embeddings/s2_train_glove_embeddings.npy
    python hf_sync.py delete-pattern "data/embeddings/*.npy"

Reads HF_TOKEN and HF_REPO_ID from .env file in the same directory.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

TOKEN   = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("HF_REPO_ID", "stat214-group08/lab3")

if not TOKEN:
    sys.exit("Error: HF_TOKEN not found. Add it to your .env file.")


def get_api():
    return HfApi(token=TOKEN)


def cmd_upload(args):
    api = get_api()
    print(f"Uploading {args.local_path} -> {REPO_ID}/{args.repo_path} ...")
    url = api.upload_file(
        path_or_fileobj=args.local_path,
        path_in_repo=args.repo_path,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Done: {url}")


def cmd_upload_folder(args):
    api = get_api()
    print(f"Uploading folder {args.local_folder} -> {REPO_ID}/{args.repo_folder} ...")
    url = api.upload_folder(
        folder_path=args.local_folder,
        path_in_repo=args.repo_folder,
        repo_id=REPO_ID,
        repo_type="dataset",
        ignore_patterns=[],  # override .gitignore — data/ is gitignored locally but must be uploaded
    )
    print(f"Done: {url}")


def cmd_download(args):
    local_path = Path(args.local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}/{args.repo_path} -> {local_path} ...")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=args.repo_path,
        repo_type="dataset",
        token=TOKEN,
        local_dir=str(local_path.parent),
        local_dir_use_symlinks=False,
    )

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != local_path.resolve():
        local_path.write_bytes(downloaded_path.read_bytes())
    print(f"Done: {local_path}")


def cmd_download_folder(args):
    api = get_api()
    local_folder = Path(args.local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)
    repo_folder = args.repo_folder.strip("/")
    allow_pattern = f"{repo_folder}/**" if repo_folder else "**"
    print(f"Downloading folder {REPO_ID}/{repo_folder or '.'} -> {local_folder} ...")

    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    prefix = f"{repo_folder}/" if repo_folder else ""
    matching = [p for p in all_files if p.startswith(prefix)] if prefix else list(all_files)
    if not matching:
        similar = [p for p in all_files if repo_folder and repo_folder.split("/")[-1] in p]
        print(f"No files found under '{repo_folder}' in {REPO_ID}.")
        if similar:
            print("Did you mean one of these prefixes?")
            shown = set()
            for path in similar:
                head = path.split("/", 1)[0]
                if head not in shown:
                    shown.add(head)
                    print(f"  - {head}")
                if len(shown) >= 8:
                    break
        else:
            print("Available top-level prefixes:")
            shown = []
            for path in all_files:
                head = path.split("/", 1)[0]
                if head not in shown:
                    shown.append(head)
            for head in shown[:12]:
                print(f"  - {head}")
        return

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        token=TOKEN,
        local_dir=str(local_folder),
        allow_patterns=[allow_pattern],
    )
    print(f"Done: {path} ({len(matching)} files matched)")


def cmd_delete(args):
    api = get_api()
    print(f"Deleting {REPO_ID}/{args.repo_path} ...")
    api.delete_file(
        path_in_repo=args.repo_path,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("Done.")


def cmd_delete_pattern(args):
    api = get_api()
    print(f"Deleting files matching '{args.pattern}' in {REPO_ID} ...")
    api.delete_files(
        repo_id=REPO_ID,
        repo_type="dataset",
        delete_patterns=[args.pattern],
    )
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Hugging Face dataset sync utility")
    sub = parser.add_subparsers(dest="command", required=True)

    p_upload = sub.add_parser("upload", help="Upload a single file")
    p_upload.add_argument("local_path", help="Local file path")
    p_upload.add_argument("repo_path", help="Path in the HF repo")

    p_folder = sub.add_parser("upload-folder", help="Upload an entire folder")
    p_folder.add_argument("local_folder", help="Local folder path")
    p_folder.add_argument("repo_folder", help="Folder path in the HF repo")

    p_download = sub.add_parser("download", help="Download a single file")
    p_download.add_argument("repo_path", help="Path in the HF repo")
    p_download.add_argument("local_path", help="Local output file path")

    p_download_folder = sub.add_parser("download-folder", help="Download all files under a repo folder")
    p_download_folder.add_argument("repo_folder", help="Folder path in the HF repo, e.g. 'data/embeddings'")
    p_download_folder.add_argument("local_folder", help="Local destination folder")

    p_delete = sub.add_parser("delete", help="Delete a single file from HF repo")
    p_delete.add_argument("repo_path", help="Path in the HF repo to delete")

    p_pattern = sub.add_parser("delete-pattern", help="Delete files matching a glob pattern")
    p_pattern.add_argument("pattern", help="Glob pattern, e.g. 'data/embeddings/*.npy'")

    args = parser.parse_args()
    {"upload": cmd_upload,
     "upload-folder": cmd_upload_folder,
        "download": cmd_download,
        "download-folder": cmd_download_folder,
     "delete": cmd_delete,
     "delete-pattern": cmd_delete_pattern}[args.command](args)


if __name__ == "__main__":
    main()
