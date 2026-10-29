"""NEW portable performance replay, NOT the harness that generated the 44 results.

Reuses the preserved quality replay in one fresh child per invocation. Includes
quality hashes/evaluation/output; its timing boundary differs from the original
checkpointed harness. No original Windows sleep/event admission rule is claimed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def worker(args):
    phases = defaultdict(lambda: dict(wall_seconds=0., cpu_seconds=0., calls=0))
    def measured(label, function):
        def call(*values, **keywords):
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                return function(*values, **keywords)
            finally:
                part = phases[label]
                part["wall_seconds"] += time.perf_counter() - wall
                part["cpu_seconds"] += time.process_time() - cpu
                part["calls"] += 1
        return call
    started = time.perf_counter()
    try:
        import psutil
        process = psutil.Process()
        if args.affinity:
            process.cpu_affinity(args.affinity)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = str(args.threads)
        spec = importlib.util.spec_from_file_location("preserved_quality", HERE / "src/evaluate.py")
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)
        begin, cpu = time.perf_counter(), time.process_time()
        import numpy as np
        import scipy
        import scipy.sparse as sp
        import torch
        phases["scientific_imports"] = dict(wall_seconds=time.perf_counter()-begin, cpu_seconds=time.process_time()-cpu, calls=1)
        q.np, q.scipy, q.sp, q.torch = np, scipy, sp, torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        namespace, segments = measured("author_hash_and_load", q.author_namespace)(args.author_source)
        for name, label in (("normalize_sparse_adjacency_matrix_sparse", "normalization"), ("csr2torch", "conversion")):
            namespace[name] = measured(label, namespace[name])
        for owner, name, label in ((sp,"load_npz","data_load"), (q,"check_data","input_validation"),
                (q,"sha","file_hash"), (q,"streaming_direct","predict_direct"),
                (q,"canonical_top","mask_rank_top20"), (q,"metrics","metrics"),
                (np,"savez_compressed","output_npz")):
            setattr(owner, name, measured(label, getattr(owner,name)))
        original_model = q.model_for
        def model_for(*values):
            model = measured("model_init", original_model)(*values)
            model.predict_batch = measured("predict_author", model.predict_batch)
            save(args.output / "graph_ready.json", {"rss_bytes":process.memory_info().rss, "monotonic_seconds":time.perf_counter(),
                "boundary":"model constructed; temporary graph-conversion objects can still be live; differs from original harness graph-ready baseline"})
            return model
        q.model_for = model_for
        if args.self_check:
            result = measured("synthetic_self_check_total", q.self_check)(namespace)
        else:
            replay_args = argparse.Namespace(**vars(args))
            replay_args.output = args.output / "quality_output"
            result = measured("quality_replay_total", q.replay)(replay_args, namespace, segments)
        info = process.memory_info()
        evidence = {"status":"complete", "synthetic_only":args.self_check, "result":result,
            "new_tool_not_original_44_harness":True,
            "worker_elapsed_to_evidence_seconds":time.perf_counter()-started,
            "phases":dict(phases), "phases_nested_do_not_sum":True,
            "phase_limits":"Score-row hashing, dense batch input creation and JSON serialization are inside total but not separately timed; model_init is inside graph construction; file_hash may be nested.",
            "environment":{"python":platform.python_version(),"platform":platform.platform(),
                "numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,"psutil":psutil.__version__,
                "torch_threads":torch.get_num_threads(),"torch_interop_threads":torch.get_num_interop_threads(),
                "cpu_affinity":process.cpu_affinity(),"device":"cpu","dtype":"float32"},
            "rss_at_evidence_bytes":info.rss,"windows_lifetime_peak_bytes":getattr(info,"peak_wset",None),
            "script_sha256":q.sha(__file__),"quality_entry_sha256":q.sha(q.__file__),
            "author_segment_sha256":segments,
            "timing_inference":"Synthetic timings are tool validation only; independent replays do not replace the original 44 measurements."}
        save(args.output / "worker_evidence.json", evidence)
        print(json.dumps({"status":"complete","synthetic_only":args.self_check}))
    except Exception:
        save(args.output / "worker_evidence.json", {"status":"failed","synthetic_only":args.self_check,
            "error":traceback.format_exc(),"phases":dict(phases),"new_tool_not_original_44_harness":True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh directory, never overwritten")
    parser.add_argument("--self-check", action="store_true", help="tiny synthetic matrices only")
    parser.add_argument("--config", choices=("yelp_f5","amazon_f2"))
    parser.add_argument("--route", choices=("author_snapshot","streaming_direct"))
    parser.add_argument("--train",type=Path)
    parser.add_argument("--test",type=Path)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--affinity",type=lambda s:[int(v) for v in s.split(",")],help="optional logical CPU IDs, e.g. 0,1,2,3; not physical-core count")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads<1 or args.batch_size<1:
        parser.error("positive threads and batch size required")
    if args.affinity and (min(args.affinity)<0 or len(set(args.affinity))!=len(args.affinity)):
        parser.error("affinity must contain unique nonnegative logical IDs")
    real_args = (args.config,args.route,args.train,args.test)
    if args.self_check and any(v is not None for v in real_args):
        parser.error("self-check cannot accept real-data arguments")
    if not args.self_check and any(v is None for v in real_args):
        parser.error("replay requires --config --route --train --test")
    args.output = args.output.resolve()
    args.author_source = args.author_source.resolve()
    if args.worker:
        return worker(args)
    if args.output.exists():
        parser.error("output exists; choose a fresh directory")
    import psutil
    args.output.mkdir(parents=True,exist_ok=False)
    command = [sys.executable,"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"]
    beginning = datetime.now(timezone.utc).isoformat()
    wall, epoch = time.perf_counter(), time.time()
    peak = lifetime = count = 0
    first = last = None
    gap = 0.
    errors = []
    with (args.output/"worker.log").open("w",encoding="utf-8") as log:
        child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        process = psutil.Process(child.pid)
        while True:
            try:
                now = time.perf_counter()
                info = process.memory_info()
                peak = max(peak,info.rss)
                lifetime = max(lifetime,getattr(info,"peak_wset",0))
                if first is None: first = now
                if last is not None: gap = max(gap,now-last)
                last,count = now,count+1
            except psutil.Error as exc:
                if len(errors)<3: errors.append(type(exc).__name__)
            if child.poll() is not None: break
            time.sleep(.01)
        ended = time.perf_counter()
    evidence_path = args.output/"worker_evidence.json"
    worker_record = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    lifetime = max(lifetime,worker_record.get("windows_lifetime_peak_bytes") or 0)
    valid = child.returncode==0 and worker_record.get("status")=="complete" and count>0
    record = {"tool":"NEW portable performance replay; not original 44 harness", "synthetic_only":args.self_check,
        "begin_utc":beginning,"end_utc":datetime.now(timezone.utc).isoformat(),"command":command,"exit_code":child.returncode,
        "launch_to_exit_wall_seconds":ended-wall,"epoch_elapsed_seconds":time.time()-epoch,
        "rss_full_lifecycle_sampled_peak_bytes":peak,"windows_lifetime_peak_bytes":lifetime or None,
        "observed_process_peak_bytes":max(peak,lifetime),"sampler":{"requested_interval_seconds":.01,"samples":count,
            "first_sample_delay_seconds":None if first is None else first-wall,"max_observed_gap_seconds":gap,
            "last_sample_to_exit_seconds":None if last is None else ended-last,"errors":errors},
        "completed_with_rss_samples":valid,
        "original_timing_admission_rules_applied":False,
        "limits":"No AC/sleep-event audit, system-awake request, cooldown, CPU topology certification or original checkpoint layout. Sampling can miss transients; Windows lifetime peak is reported when available. New replay boundary includes validation, digest/evaluation and output.",
        "formal_44_result":False}
    save(args.output/"performance.json",record)
    print(json.dumps({"completed_with_rss_samples":valid,"synthetic_only":args.self_check,"output":str(args.output)}))
    if not valid: raise SystemExit(1)


if __name__=="__main__":
    main()
