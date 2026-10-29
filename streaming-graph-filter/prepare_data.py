"""Recreate the four NPZ inputs from separately held pinned LightGCN text.

Uses the recovered original data loader and _pos_to_csr function. No remapping,
resplitting, downloads or model execution. Writes only to a fresh output folder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def sparse_content_digest(matrix):
    h=hashlib.sha256()
    h.update(json.dumps(list(matrix.shape)).encode('ascii'))
    for array in (matrix.indptr,matrix.indices,matrix.data):
        # Canonical little-endian/int64 indices separate semantic structure from ZIP bytes.
        value=array.astype('<f8' if array is matrix.data else '<i8',copy=False)
        h.update(value.tobytes())
    return h.hexdigest()


def run(raw_root,output,processed_root):
    import numpy as np
    import scipy
    import scipy.sparse as sp
    sys.path.insert(0,str(ROOT))
    from src.data import load_lightgcn_format
    from src.preprocessing import _pos_to_csr
    raw=json.loads((ROOT/'configs/raw_inputs.json').read_text(encoding='utf-8'))['files']
    reference=json.loads((ROOT/'configs/inputs.json').read_text(encoding='utf-8'))['inputs']
    for dataset in ('yelp2018','amazon-book'):
        for filename in ('train.txt','test.txt'):
            name=f'{dataset}/{filename}'
            path=raw_root/name
            assert path.stat().st_size==raw[name]['bytes'] and sha(path)==raw[name]['sha256'],name
    output.mkdir(parents=True,exist_ok=False)
    records={}
    for dataset in ('yelp2018','amazon-book'):
        bundle=load_lightgcn_format(str(raw_root/dataset),dataset)
        matrices={'R_train_full_official.npz':bundle.R_train,'R_test.npz':_pos_to_csr(bundle.test_pos,bundle.R_train.shape)}
        (output/dataset).mkdir()
        for filename,matrix in matrices.items():
            name=f'{dataset}/{filename}'
            dest=output/name
            sp.save_npz(dest,matrix)
            saved=sp.load_npz(dest).tocsr()
            record={'shape':list(saved.shape),'nnz':saved.nnz,'dtype':str(saved.dtype),'binary':bool(np.all(saved.data==1)),
                'canonical_csr':bool(saved.has_canonical_format),'zero_rows':int(np.count_nonzero(np.diff(saved.indptr)==0)),
                'train_test_overlap_not_checked_by_this_function':True,
                'sha256':sha(dest),'bytes':dest.stat().st_size,'sparse_content_sha256':sparse_content_digest(saved),
                'frozen_npz_sha256':reference[name]['sha256'],'generated_npz_bytes_match_frozen':sha(dest)==reference[name]['sha256']}
            if processed_root:
                original=sp.load_npz(processed_root/name).tocsr()
                equal=saved.shape==original.shape and all(np.array_equal(a,b) for a,b in zip((saved.indptr,saved.indices,saved.data),(original.indptr,original.indices,original.data)))
                record['frozen_input_sha256_reverified']=sha(processed_root/name)==reference[name]['sha256']
                record['semantic_arrays_equal_to_frozen']=bool(equal)
                assert equal and record['frozen_input_sha256_reverified'],name
            records[name]=record
    report={'records':records,'raw_text_hashes_checked':4,'recovered_loader_used':True,'new_validation_split_used':False,
        'original_review_event_to_lightgcn_split_reproduced':False,'model_scoring_performed':False,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'limits':'Pinned processed text -> full official train/test only. Original download commit, original event-to-ID mapping and original split construction remain unverified; equal sparse arrays and equal NPZ bytes are reported separately.'}
    (output/'reconstruction_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return {'matrices':len(records),'generated_npz_hash_matches':sum(r['generated_npz_bytes_match_frozen'] for r in records.values()),'semantic_matches':sum(r.get('semantic_arrays_equal_to_frozen',False) for r in records.values()),'model_scoring_performed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--processed-root',type=Path,help='optional authorized originals for exact sparse-array comparison')
    args=parser.parse_args()
    print(json.dumps(run(args.raw_root,args.output,args.processed_root)))
