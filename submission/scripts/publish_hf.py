"""Upload staged folders to Hugging Face.

Each sub-folder of --stage is named `<owner>__<repo>` and is uploaded to `<owner>/<repo>`.
The token is read from --token-file (default ~/.ra_hf_token) or the HF_TOKEN environment variable;
nothing is uploaded without one. New repos are private unless --public is given.

    python submission/scripts/publish_hf.py --stage /var/tmp/sra_publish
"""

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--token-file", default=str(Path.home() / ".ra_hf_token"))
    ap.add_argument("--public", action="store_true", help="create NEW repos as public")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    tf = Path(args.token_file)
    if not token and tf.exists():
        token = tf.read_text().strip()
    if not token:
        raise SystemExit(f"no Hugging Face token (set HF_TOKEN or create {tf}); nothing uploaded")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print("uploading as", api.whoami()["name"])
    for d in sorted(Path(args.stage).iterdir()):
        if not d.is_dir() or "__" not in d.name:
            continue
        owner, name = d.name.split("__", 1)
        repo = f"{owner}/{name}"
        api.create_repo(repo, repo_type="model", exist_ok=True, private=not args.public)
        api.upload_folder(
            folder_path=str(d),
            repo_id=repo,
            repo_type="model",
            commit_message="Automated upload from the training pipeline",
        )
        print("uploaded", repo)


if __name__ == "__main__":
    main()
