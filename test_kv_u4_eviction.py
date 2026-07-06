#!/usr/bin/env python3
"""Verify KV_CACHE_PRECISION=u4 and cache eviction on local GPU."""
import os
import sys
import time

import openvino_genai as ov_genai


def mem_available_mib() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return -1


def build_scheduler(n_ctx: int = 32768) -> ov_genai.SchedulerConfig:
    scheduler = ov_genai.SchedulerConfig()
    scheduler.max_num_batched_tokens = 128
    scheduler.num_kv_blocks = n_ctx // 16
    scheduler.dynamic_split_fuse = True
    scheduler.enable_prefix_caching = True
    return scheduler


def build_scheduler_with_eviction(n_ctx: int = 32768) -> ov_genai.SchedulerConfig:
    scheduler = build_scheduler(n_ctx)
    # apply_rotation=False required when using INT4 KV cache (OpenVINO 2026.2)
    scheduler.cache_eviction_config = ov_genai.CacheEvictionConfig(
        start_size=32,
        recent_size=128,
        max_cache_size=2048,
        aggregation_mode=ov_genai.AggregationMode.NORM_SUM,
        apply_rotation=False,
        snapkv_window_size=8,
        kvcrush_config=ov_genai.KVCrushConfig(
            budget=2,
            anchor_point_mode=ov_genai.KVCrushAnchorPointMode.MEAN,
        ),
    )
    scheduler.use_cache_eviction = True
    return scheduler


def run_case(
    label: str,
    model_dir: str,
    device: str,
    scheduler: ov_genai.SchedulerConfig,
    kv_precision: str | None,
) -> bool:
    print(f"\n{'=' * 60}")
    print(f"CASE: {label}")
    print(f"  kv_precision={kv_precision!r}, eviction={scheduler.use_cache_eviction}")
    print(f"  num_kv_blocks={scheduler.num_kv_blocks}, mem_before={mem_available_mib()} MiB")

    props = {}
    if kv_precision:
        props["KV_CACHE_PRECISION"] = kv_precision

    try:
        t0 = time.time()
        pipe = ov_genai.ContinuousBatchingPipeline(model_dir, scheduler, device, props)
        load_s = time.time() - t0
        print(f"  LOAD OK in {load_s:.1f}s, mem_after_load={mem_available_mib()} MiB")

        prompt = "Explain binary search in one paragraph."
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 64

        handle = pipe.add_request(0, prompt, config)
        tokens = 0
        t1 = time.time()
        while True:
            pipe.step()
            while handle.can_read():
                res = handle.read()
                for out in res.values():
                    tokens += len(out.generated_ids)
            status = handle.get_status()
            if status == ov_genai.GenerationStatus.FINISHED:
                break
            if time.time() - t1 > 120:
                print("  GENERATE TIMEOUT")
                return False

        gen_s = time.time() - t1
        try:
            metrics = pipe.get_metrics()
            print(
                f"  GENERATE OK: {tokens} tokens in {gen_s:.1f}s, "
                f"cache_usage={metrics.cache_usage:.3f}, "
                f"mem_after={mem_available_mib()} MiB"
            )
        except Exception:
            print(f"  GENERATE OK: {tokens} tokens in {gen_s:.1f}s, mem_after={mem_available_mib()} MiB")

        del pipe
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/DeepSeek-R1-Distill-Qwen-1.5B-int8-ov"
    device = sys.argv[2] if len(sys.argv) > 2 else "GPU"
    n_ctx = 32768

    if device == "GPU":
        os.environ["OV_GPU_FP16_SKIP_OPTIMIZATION"] = "1"
        os.environ["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "1"
        os.environ["OV_GPU_WAIT_TYPE"] = "SLEEP"

    print(f"Model: {model_dir}, device: {device}, n_ctx: {n_ctx}")

    results = {}

    # Default INT8 KV (implicit)
    results["default_u8"] = run_case(
        "default (implicit INT8 KV)",
        model_dir,
        device,
        build_scheduler(n_ctx),
        kv_precision=None,
    )

    # Explicit INT4 KV
    results["u4_kv"] = run_case(
        "KV_CACHE_PRECISION=u4",
        model_dir,
        device,
        build_scheduler(n_ctx),
        kv_precision="u4",
    )

    # INT4 KV + cache eviction + KVCrush
    results["u4_eviction"] = run_case(
        "KV_CACHE_PRECISION=u4 + eviction + KVCrush",
        model_dir,
        device,
        build_scheduler_with_eviction(n_ctx),
        kv_precision="u4",
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
