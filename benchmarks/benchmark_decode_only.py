#!/usr/bin/env python3
"""Benchmark steady-state *decode* latency of a single batch of requests.

This script measures the inter-token latency (ITL) and the time-per-output-token
(TPOT) *after* all prefills have completed.  It sends a user-specified batch of
requests at once (request_rate = ∞) so that the serving backend prefills them
together and then enters the decode phase.  By default we ignore the first
``skip_initial_tokens`` tokens from every request to ensure we are firmly in the
steady-state decode phase before measuring.

Example usage (vLLM OpenAI-compatible server running on localhost:8000):

```bash
python benchmarks/benchmark_decode_only.py \
  --backend openai-chat \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --batch-size 32 \
  --input-len 1024 \
  --output-len 128
```
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from typing import Iterable, List, Optional

import numpy as np

# Re-use the existing helper utilities from the main benchmark suite.
from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
    get_tokenizer,
)
from benchmark_dataset import RandomDataset, SampleRequest

MILLISECONDS = 1000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure steady-state decode latency (ITL/TPOT) for a single batch of requests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------------------------------------------------
    # Server connection parameters
    # ---------------------------------------------------------------------
    parser.add_argument("--backend", type=str, default="openai-chat", choices=list(ASYNC_REQUEST_FUNCS.keys()))
    parser.add_argument("--model", type=str, required=True, help="Name of the model as exposed by the server (e.g. `/my/org/model`).")
    parser.add_argument("--served-model-name", type=str, default=None, help="If the server registers a different model name in the API, specify it here.")

    group_url = parser.add_mutually_exclusive_group()
    group_url.add_argument("--base-url", type=str, help="Full base URL of the server, e.g. http://127.0.0.1:8000")
    group_url.add_argument("--host", type=str, default="127.0.0.1", help="Server host (ignored if --base-url is provided)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (ignored if --base-url is provided)")
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions", help="API endpoint relative to base URL.")

    # ---------------------------------------------------------------------
    # Workload parameters
    # ---------------------------------------------------------------------
    parser.add_argument("--batch-size", type=int, default=32, help="Number of requests to send simultaneously (forms one prefill batch).")
    parser.add_argument("--input-len", type=int, default=1024, help="Number of *prompt* tokens per request.")
    parser.add_argument("--output-len", type=int, default=128, help="Number of tokens to decode per request.")

    parser.add_argument("--skip-initial-tokens", type=int, default=2, help="Number of first decoded tokens to discard from latency stats (per request).")

    # Repetition / warm-up parameters
    parser.add_argument("--warmup-batches", type=int, default=1, help="Number of warm-up batches to run before measurement.")
    parser.add_argument("--measure-batches", type=int, default=5, help="Number of batches to measure.")
    parser.add_argument("--between-batch-sleep", type=float, default=0.0, help="Seconds to sleep between consecutive batches.")

    # Sampling parameters (only for OpenAI-compatible backends)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)

    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic prompt generation.")

    parser.add_argument("--report-batch-iterations", action="store_true", help="Report batch-level decode iteration latency (max ITL across requests).")

    return parser.parse_args()


async def _run_batch(
    backend: str,
    api_url: str,
    model_id: str,
    model_name: str,
    tokenizer,
    input_requests: List[SampleRequest],
    sampling_params: dict[str, float | int | bool],
) -> List[RequestFuncOutput]:
    """Fire off *all* requests concurrently and collect streaming stats."""

    request_func = ASYNC_REQUEST_FUNCS[backend]

    tasks = []
    for req in input_requests:
        body = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=req.prompt,
            api_url=api_url,
            prompt_len=req.prompt_len,
            output_len=req.expected_output_len,
            logprobs=None,
            multi_modal_content=req.multi_modal_data,
            ignore_eos=False,
            extra_body=sampling_params if backend in OPENAI_COMPATIBLE_BACKENDS else None,
        )
        tasks.append(asyncio.create_task(request_func(request_func_input=body)))

    outputs = await asyncio.gather(*tasks)
    return list(outputs)


def _aggregate_latency(
    outputs: Iterable[RequestFuncOutput],
    skip_initial_tokens: int,
) -> dict[str, float]:
    """Compute ITL/TPOT statistics after skipping the first *skip_initial_tokens* per request."""

    all_itls = []
    per_request_tpot = []  # average ITL per request (after skipping)

    for out in outputs:
        if not out.success:
            continue
        itls = out.itl[skip_initial_tokens:]
        if not itls:
            continue  # nothing to measure
        all_itls.extend(itls)
        per_request_tpot.append(float(np.mean(itls)))

    if not all_itls:
        raise RuntimeError("No inter-token latencies recorded. Ensure that the backend supports streaming and that output_len > skip_initial_tokens + 1.")

    stats: dict[str, float] = {}
    itls_ms = np.array(all_itls) * MILLISECONDS
    stats["mean_itl_ms"] = float(np.mean(itls_ms))
    stats["median_itl_ms"] = float(np.median(itls_ms))
    stats["p99_itl_ms"] = float(np.percentile(itls_ms, 99))
    stats["std_itl_ms"] = float(np.std(itls_ms))

    tpot_ms = np.array(per_request_tpot) * MILLISECONDS
    stats["mean_tpot_ms"] = float(np.mean(tpot_ms))
    stats["median_tpot_ms"] = float(np.median(tpot_ms))
    stats["std_tpot_ms"] = float(np.std(tpot_ms))

    return stats


def _build_requests(
    tokenizer,
    batch_size: int,
    input_len: int,
    output_len: int,
    seed: int,
) -> List[SampleRequest]:
    random.seed(seed)
    np.random.seed(seed)

    dataset = RandomDataset(dataset_path=None)
    return dataset.sample(
        tokenizer=tokenizer,
        num_requests=batch_size,
        input_len=input_len,
        output_len=output_len,
        range_ratio=0.0,
        prefix_len=0,
    )


async def main_async(args: argparse.Namespace):
    # ------------------------------------------------------------------
    # Resolve connection details
    # ------------------------------------------------------------------
    if args.base_url:
        base_url = args.base_url.rstrip("/")
    else:
        base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}{args.endpoint}"

    # ------------------------------------------------------------------
    # Prepare tokenizer and synthetic prompts
    # ------------------------------------------------------------------
    tokenizer = get_tokenizer(args.model, tokenizer_mode="auto", trust_remote_code=True)

    requests = _build_requests(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        input_len=args.input_len,
        output_len=args.output_len,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Collect decode-only latency data
    # ------------------------------------------------------------------
    sampling_params = {
        k: v
        for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature,
        }.items()
        if v is not None
    }

    # ------------------------------------------------------------------
    # Warm-up batches (not measured)
    # ------------------------------------------------------------------
    for i in range(args.warmup_batches):
        await _run_batch(
            backend=args.backend,
            api_url=api_url,
            model_id=args.model,
            model_name=args.served_model_name or args.model,
            tokenizer=tokenizer,
            input_requests=requests,
            sampling_params=sampling_params,
        )
        if args.between_batch_sleep > 0:
            await asyncio.sleep(args.between_batch_sleep)

    # ------------------------------------------------------------------
    # Measured batches
    # ------------------------------------------------------------------
    measured_itls = []
    measured_tpots = []
    measured_iteration_latencies = []
    total_wall_time = 0.0
    total_success = 0

    for batch_idx in range(args.measure_batches):
        st_batch = time.perf_counter()
        outputs = await _run_batch(
            backend=args.backend,
            api_url=api_url,
            model_id=args.model,
            model_name=args.served_model_name or args.model,
            tokenizer=tokenizer,
            input_requests=requests,
            sampling_params=sampling_params,
        )
        total_wall_time += time.perf_counter() - st_batch
        total_success += sum(o.success for o in outputs)

        # Gather latency lists
        for o in outputs:
            if not o.success:
                continue
            itls = o.itl[args.skip_initial_tokens:]
            if not itls:
                continue
            measured_itls.extend(itls)
            measured_tpots.append(float(np.mean(itls)))

        # Collect batch-level iteration latency (max over requests for each decode step)
        if args.report_batch_iterations:
            # Build a ragged matrix of itl lists for this batch
            batch_itl_lists = [
                o.itl[args.skip_initial_tokens:]
                for o in outputs
                if o.success and len(o.itl) > args.skip_initial_tokens
            ]
            if batch_itl_lists:
                max_len = max(len(lst) for lst in batch_itl_lists)
                for idx in range(max_len):
                    vals = [lst[idx] for lst in batch_itl_lists if len(lst) > idx]
                    if vals:
                        measured_iteration_latencies.append(max(vals))

        if args.between_batch_sleep > 0 and batch_idx != args.measure_batches - 1:
            await asyncio.sleep(args.between_batch_sleep)

    if not measured_itls:
        raise RuntimeError("No successful requests or ITLs recorded during measured batches.")

    itls_ms = np.array(measured_itls) * MILLISECONDS
    tpot_ms = np.array(measured_tpots) * MILLISECONDS

    stats = {
        "mean_itl_ms": float(np.mean(itls_ms)),
        "median_itl_ms": float(np.median(itls_ms)),
        "p99_itl_ms": float(np.percentile(itls_ms, 99)),
        "std_itl_ms": float(np.std(itls_ms)),
        "mean_tpot_ms": float(np.mean(tpot_ms)),
        "median_tpot_ms": float(np.median(tpot_ms)),
        "std_tpot_ms": float(np.std(tpot_ms)),
    }

    if args.report_batch_iterations and measured_iteration_latencies:
        iter_ms = np.array(measured_iteration_latencies) * MILLISECONDS
        batch_iter_stats = {
            "mean_batch_iter_ms": float(np.mean(iter_ms)),
            "median_batch_iter_ms": float(np.median(iter_ms)),
            "p99_batch_iter_ms": float(np.percentile(iter_ms, 99)),
            "std_batch_iter_ms": float(np.std(iter_ms)),
        }
        stats.update(batch_iter_stats)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("".center(60, "="))
    print(" Steady-state Decode Latency Benchmark ".center(60, "="))
    print("".center(60, "="))
    print(f"Backend                 : {args.backend}")
    print(f"Model                   : {args.model}")
    print(f"Batch size              : {args.batch_size}")
    print(f"Prompt tokens / request : {args.input_len}")
    print(f"Output tokens / request : {args.output_len}")
    print(f"Skip initial tokens     : {args.skip_initial_tokens}")
    print(f"Warm-up batches         : {args.warmup_batches}")
    print(f"Measured batches        : {args.measure_batches}")
    print(f"Requests completed      : {total_success} / {args.measure_batches * len(requests)}")
    print(f"Wall-clock time (s)     : {total_wall_time:.3f}")
    print("-" * 60)
    print(f"Mean ITL (ms)           : {stats['mean_itl_ms']:.2f}")
    print(f"Median ITL (ms)         : {stats['median_itl_ms']:.2f}")
    print(f"P99 ITL (ms)            : {stats['p99_itl_ms']:.2f}")
    print(f"Std-dev ITL (ms)        : {stats['std_itl_ms']:.2f}")
    print("-")
    print(f"Mean TPOT (ms)          : {stats['mean_tpot_ms']:.2f}")
    print(f"Median TPOT (ms)        : {stats['median_tpot_ms']:.2f}")
    print(f"Std-dev TPOT (ms)       : {stats['std_tpot_ms']:.2f}")
    if args.report_batch_iterations and 'mean_batch_iter_ms' in stats:
        print("-")
        print(f"Mean batch iter (ms)    : {stats['mean_batch_iter_ms']:.2f}")
        print(f"Median batch iter (ms)  : {stats['median_batch_iter_ms']:.2f}")
        print(f"P99 batch iter (ms)     : {stats['p99_batch_iter_ms']:.2f}")
        print(f"Std-dev batch iter (ms) : {stats['std_batch_iter_ms']:.2f}")
        tokens_per_sec = args.batch_size * 1000.0 / stats['mean_batch_iter_ms']
        print("-")
        print(f"Token throughput (tok/s): {tokens_per_sec:.1f}")
    print("=" * 60)


def main():
    args = _parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()