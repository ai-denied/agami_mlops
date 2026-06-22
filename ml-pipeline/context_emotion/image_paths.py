"""Resolve emotic/manual rows to an actual, openable image file on disk.

The raw EMOTIC data on disk is an incomplete, multiply-duplicated mirror
(see RECONSTRUCTION_NOTES.md) - the same sub-dataset can have up to three
candidate directories, and not every annotated image was ever downloaded.
This module tries each known candidate location in a fixed priority order
and verifies the file actually decodes (catches the "truncated copy"
problem the Day 4 retrospective described for framesdb).
"""
import os

from PIL import Image, UnidentifiedImageError

EMOTIC_ROOT_A = "/workspace/data/context_emotion/emotic"
EMOTIC_ROOT_B = "/workspace/data/context_emotion/emotic_dataset/emotic"
MANUAL_ROOT = "/workspace/data/context_emotion/manual_images"

# folder (as it appears in Annotations.mat) -> ordered list of candidate dirs
EMOTIC_CANDIDATE_DIRS = {
    "mscoco/images": [
        f"{EMOTIC_ROOT_A}/mscoco/mscoco/images",
        f"{EMOTIC_ROOT_B}/mscoco/images",
        f"{EMOTIC_ROOT_A}/mscoco/images",
    ],
    "framesdb/images": [
        f"{EMOTIC_ROOT_A}/framesdb/framesdb/images",
        f"{EMOTIC_ROOT_B}/framesdb/images",
        f"{EMOTIC_ROOT_A}/framesdb/images",
    ],
    "emodb_small/images": [
        f"{EMOTIC_ROOT_B}/emodb_small/images",
    ],
    "ade20k/images": [
        f"{EMOTIC_ROOT_B}/ade20k/images",
    ],
}


def _verify_decodes(path):
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def resolve_emotic_path(folder, filename):
    for base in EMOTIC_CANDIDATE_DIRS.get(folder, []):
        candidate = os.path.join(base, filename)
        if os.path.isfile(candidate) and _verify_decodes(candidate):
            return candidate
    return None


def resolve_manual_path(folder, filename):
    candidate = os.path.join(MANUAL_ROOT, folder, filename)
    if os.path.isfile(candidate) and _verify_decodes(candidate):
        return candidate
    return None


def inspect_image(path):
    """Width/height + two hashes for an already-resolved, already-decodable
    image path: content_hash (exact-duplicate detection across the
    mirrored/duplicated EMOTIC directories) and perceptual_hash (catches
    re-encoded/resized near-duplicates that have a different content_hash)."""
    import hashlib

    import imagehash

    with open(path, "rb") as f:
        data = f.read()
    content_hash = hashlib.sha256(data).hexdigest()

    with Image.open(path) as img:
        width, height = img.size
        perceptual_hash = str(imagehash.average_hash(img))

    return {
        "width": width,
        "height": height,
        "content_hash": content_hash,
        "perceptual_hash": perceptual_hash,
    }
