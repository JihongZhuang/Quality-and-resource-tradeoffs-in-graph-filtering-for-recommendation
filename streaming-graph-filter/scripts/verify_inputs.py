"""Hash-check separately obtained inputs/source with stdlib; never loads matrices."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):
            digest.update(part)
    return digest.hexdigest()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--author-source",type=Path)
    args=parser.parse_args()
    if args.data_root is None and args.author_source is None:
        parser.error("provide --data-root and/or --author-source")
    records=[]
    if args.data_root is not None:
        data=json.loads((args.bundle/"configs/inputs.json").read_text(encoding="utf-8"))
        records += [(args.data_root/name,record) for name,record in data["inputs"].items()]
    if args.author_source is not None:
        quality=json.loads((args.bundle/"configs/reference.json").read_text(encoding="utf-8"))
        records += [(args.author_source/name,record) for name,record in quality["author_snapshot"]["files"].items()]
    for path,record in records:
        assert path.stat().st_size==record["bytes"] and sha(path)==record["sha256"],str(path)
    print(json.dumps({"verified_external_files":len(records),"matrix_loading_or_scoring":False}))
