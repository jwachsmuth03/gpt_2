"""Shared, deterministic language-model corpus split for Assignments 2 and 3."""
import hashlib
from pathlib import Path
from urllib.request import urlopen
from bpe import normalize

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PTB_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
SHAKESPEARE_SHA256 = "434c0554a8c4c53dc17e56a0abb0f30b88f83cbceb0289cb897db68c25e89eba"
PTB_MD5 = "8b80168b89c18661a38ef683c0dc3721"


def download_if_missing(path, url):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
    return path


def load_shakespeare():
    path = Path("data/shakespeare_corpus.txt")
    if not path.exists() and Path("shakespeare_corpus.txt").exists():
        path = Path("shakespeare_corpus.txt")
    path = download_if_missing(path, SHAKESPEARE_URL)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHAKESPEARE_SHA256:
        raise ValueError("Shakespeare differs from the supplied corpus; check the data source.")
    lines = [normalize(line) for line in path.read_text(encoding="utf-8").splitlines()]
    first, second = int(0.8 * len(lines)), int(0.9 * len(lines))
    return {"train": lines[:first], "val": lines[first:second], "test": lines[second:]}


def write_split_files(lines, directory="data/assignment2"):
    """Persist one clean line per sequence, including empty lines."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, part in lines.items():
        path = directory / f"{name}.txt"
        path.write_text("\n".join(part) + "\n", encoding="utf-8")
        assert path.read_text(encoding="utf-8").splitlines() == part
        paths[name] = path
    return paths
