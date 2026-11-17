"""Upload NeuroLens to a Hugging Face Space.

    python deploy_space.py              # browser version (web/) as a private static Space "neurolens"
    python deploy_space.py my-demo      # different Space name
    python deploy_space.py --public     # visible to everyone
    python deploy_space.py --gradio     # the Gradio app instead (needs Hugging Face PRO)

The browser version runs the model client-side, so it fits Hugging Face's free static Spaces.
The first run asks you to log in. Re-running updates the existing Space in place. Visibility is
only set when the Space is first created; after that, change it under Settings on the Space page.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, get_token, login

HERE = Path(__file__).parent
GRADIO_IGNORE = ["__pycache__/*", "*.pyc", ".claude/*", "deploy_space.py", ".venv/*", "tools/*", "README.md",
                 "web/model/*", "web/README.md", "web/index.html", "web/app.js", "web/styles.css"]
# Space card for the Gradio version (the static version's card is web/README.md)
GRADIO_CARD = """---
title: NeuroLens
emoji: 🧠
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Brain MRI tumor classifier with an interactive 3D brain
---

Brain MRI classifier (ResNet18) with Grad-CAM mapped onto an interactive 3D brain. Research demo, not a medical device.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", default="neurolens", help="Space name, or user/name")
    parser.add_argument("--public", action="store_true", help="make the Space public (default: private)")
    parser.add_argument("--gradio", action="store_true", help="deploy the Gradio app (needs PRO)")
    args = parser.parse_args()

    if not get_token():
        login()

    api = HfApi()
    repo_id = args.name if "/" in args.name else f"{api.whoami()['name']}/{args.name}"
    sdk = "gradio" if args.gradio else "static"
    folder = HERE if args.gradio else HERE / "web"

    exists = api.repo_exists(repo_id, repo_type="space")
    if exists:
        print(f"Updating {sdk} Space {repo_id}...")
    else:
        print(f"Creating {sdk} Space {repo_id} ({'public' if args.public else 'private'})...")
        api.create_repo(repo_id, repo_type="space", space_sdk=sdk, private=not args.public)

    print("Uploading files (the 45 MB model takes a moment)...")
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=folder,
                      ignore_patterns=GRADIO_IGNORE if args.gradio else ["__pycache__/*"],
                      # static: mirror web/ exactly, so template or stale files on the Space get removed
                      delete_patterns=None if args.gradio else ["*"],
                      commit_message="Deploy NeuroLens")
    if args.gradio:
        api.upload_file(path_or_fileobj=GRADIO_CARD.encode(), path_in_repo="README.md", repo_id=repo_id,
                        repo_type="space", commit_message="Space card")

    print(f"\nDone: https://huggingface.co/spaces/{repo_id}")
    if api.space_info(repo_id).private:
        print("It's private. Once it works, switch it to public under Settings on that page.")


if __name__ == "__main__":
    main()
