#!/usr/bin/env python3
"""Log in to Hugging Face and download one Hub file to an exact local path."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, login, whoami

TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def get_environment_token() -> str | None:
    for name in TOKEN_ENV_NAMES:
        token = os.environ.get(name)
        if token:
            return token
    return None


def print_identity(identity: dict[str, object]) -> None:
    
    username = identity.get("name") or identity.get("fullname") or "unknown"
    print(f"Logged in to Hugging Face as: {username}")


def login_command(args: argparse.Namespace) -> int:
    token = get_environment_token()
    if token:
        login(token=token, add_to_git_credential=args.add_to_git_credential)
    else:
        print("HF_TOKEN is not set; enter the token in the secure prompt.")
        login(add_to_git_credential=args.add_to_git_credential)
    print_identity(whoami())
    return 0


def status_command(_: argparse.Namespace) -> int:
    print_identity(whoami(token=get_environment_token()))
    return 0


def download_command(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --force to replace it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    token = get_environment_token()

    # Download into the destination filesystem, then atomically move the completed
    # file into place. An interrupted download never leaves a partial output file.
    with tempfile.TemporaryDirectory(
        prefix=".hf-download-",
        dir=output.parent,
    ) as temporary_directory:
        downloaded = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=args.filename,
                repo_type=args.repo_type,
                revision=args.revision,
                token=token,
                local_dir=temporary_directory,
                force_download=args.force,
            )
        )

        if downloaded.is_symlink():
            temporary_output = output.with_name(f".{output.name}.tmp")
            shutil.copy2(downloaded, temporary_output)
            os.replace(temporary_output, output)
        else:
            os.replace(downloaded, output)

    print(f"Downloaded hf://{args.repo_id}/{args.filename}")
    print(f"Saved to: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate with Hugging Face or download a model/dataset file. "
            "Authentication uses HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or a token "
            "saved by the login command."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser(
        "login",
        help="Save an environment token, or prompt securely for one.",
    )
    login_parser.add_argument(
        "--add-to-git-credential",
        action="store_true",
        help="Also save the token through the configured Git credential helper.",
    )
    login_parser.set_defaults(handler=login_command)

    status_parser = subparsers.add_parser(
        "status",
        help="Show the currently authenticated Hugging Face account.",
    )
    status_parser.set_defaults(handler=status_command)

    download_parser = subparsers.add_parser(
        "download",
        help="Download one Hub file to an exact local path.",
    )
    download_parser.add_argument("--repo-id", required=True, help="owner/repository")
    download_parser.add_argument(
        "--filename",
        required=True,
        help="File path inside the Hub repository.",
    )
    download_parser.add_argument(
        "--output",
        required=True,
        help="Exact local output path (the filename may be renamed).",
    )
    download_parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="model",
    )
    download_parser.add_argument("--revision", default="main")
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and replace an existing output file.",
    )
    download_parser.set_defaults(handler=download_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        print(
            "For private or gated repositories, first accept the repository's "
            "access terms and log in with a token that has read permission.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
