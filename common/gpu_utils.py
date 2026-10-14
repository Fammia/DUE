import os
import subprocess


def parse_visible_gpu_set():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return raw, None, False
    try:
        visible = {int(token.strip()) for token in raw.split(",") if token.strip()}
        return raw, visible, False
    except ValueError:
        return raw, set(), True


def _query_gpus_with_metric(metric):
    res = subprocess.run(
        ["nvidia-smi", f"--query-gpu=index,{metric}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    candidates = []
    for line in res.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            val = int(parts[1].rstrip("%").strip())
        except ValueError:
            continue
        candidates.append((idx, val))
    return candidates


def _filter_by_visible_and_exclude(candidates, exclude_gpu_ids=(), strategy_name=""):
    raw, visible_gpu_set, parse_visible_failed = parse_visible_gpu_set()
    if parse_visible_failed:
        print(
            "CUDA_VISIBLE_DEVICES parse failed in %s strategy: raw=%r; return none."
            % (strategy_name, raw)
        )
        return [], raw, visible_gpu_set, True
    exclude_set = set(exclude_gpu_ids) if exclude_gpu_ids else set()
    gpus = [(i, m) for i, m in candidates if i not in exclude_set]
    if visible_gpu_set is not None:
        gpus = [(i, m) for i, m in gpus if i in visible_gpu_set]
    return gpus, raw, visible_gpu_set, False


def get_gpu_id_str_sorted(exclude_gpu_ids=(), strategy="util"):
    if strategy not in ("mem", "util"):
        raise ValueError("strategy must be 'mem' or 'util'")

    if strategy == "mem":
        candidates = _query_gpus_with_metric("memory.free")
        gpus, raw, visible_gpu_set, failed = _filter_by_visible_and_exclude(
            candidates, exclude_gpu_ids, "mem"
        )
        if failed:
            return ""
        gpu_ids = ",".join(str(i) for i, _ in sorted(gpus, key=lambda x: x[1], reverse=True))
        print(
            "GPU IDs sorted by free memory (exclude %s): %s"
            % (exclude_gpu_ids or "none", gpu_ids or "none")
        )
    else:
        candidates = _query_gpus_with_metric("utilization.gpu")
        gpus, raw, visible_gpu_set, failed = _filter_by_visible_and_exclude(
            candidates, exclude_gpu_ids, "util"
        )
        if failed:
            return ""
        gpu_ids = ",".join(str(i) for i, _ in sorted(gpus, key=lambda x: x[1], reverse=False))
        print(
            "GPU IDs sorted by GPU utilization (ascending, exclude %s): %s"
            % (exclude_gpu_ids or "none", gpu_ids or "none")
        )

    print(
        "CUDA_VISIBLE_DEVICES(raw=%r, parsed=%s)"
        % (raw, sorted(visible_gpu_set) if visible_gpu_set is not None else "all")
    )
    return gpu_ids
