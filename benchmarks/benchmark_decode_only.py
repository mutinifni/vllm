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
import json
import os
import random
import time
from datetime import datetime
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
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json

# Import shared functions from benchmark_serving.py instead of duplicating
from benchmark_serving import (
    check_goodput_args,
    parse_goodput,
    save_to_pytorch_benchmark_format,
    create_argument_parser,
    MILLISECONDS_TO_SECONDS_CONVERSION,
)

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

MILLISECONDS = 1000.0


def _parse_args() -> argparse.Namespace:
    # Start with the shared argument parser from benchmark_serving.py
    parser = create_argument_parser()

    # Update description for decode-only
    parser.description = "Measure steady-state decode latency (ITL/TPOT) for a single batch of requests."

    # Modify defaults for decode-only specific behavior
    parser.set_defaults(
        backend="openai-chat",  # Different default than benchmark_serving
        endpoint="/v1/chat/completions",  # Different default than benchmark_serving
        dataset_name="random",  # decode_only only supports random
    )

    # Override dataset choices to only allow random for decode_only
    for action in parser._actions:
        if action.dest == 'dataset_name':
            action.choices = ["random"]
            action.help = "Name of the dataset to benchmark on. decode_only only supports 'random'."
        elif action.dest == 'request_rate':
            action.help = "Number of requests per second. For decode_only, this is always inf (all requests sent at once to form prefill batches)."
        elif action.dest == 'num_prompts':
            action.help = "Number of prompts to process. For decode_only, this is computed as batch-size * measure-batches."

    # Add decode-only specific arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Number of requests to send simultaneously (forms one prefill batch).")
    parser.add_argument("--input-len", type=int, default=1024, help="Number of *prompt* tokens per request.")
    parser.add_argument("--output-len", type=int, default=128, help="Number of tokens to decode per request.")
    parser.add_argument("--skip-initial-tokens", type=int, default=2, help="Number of first decoded tokens to discard from latency stats (per request).")

    # Repetition / warm-up parameters
    parser.add_argument("--warmup-batches", type=int, default=1, help="Number of warm-up batches to run before measurement.")
    parser.add_argument("--measure-batches", type=int, default=5, help="Number of batches to measure.")
    parser.add_argument("--between-batch-sleep", type=float, default=0.0, help="Seconds to sleep between consecutive batches.")

    return parser.parse_args()


async def _run_batch(
    backend: str,
    api_url: str,
    model_id: str,
    model_name: str,
    tokenizer,
    input_requests: List[SampleRequest],
    sampling_params: dict[str, float | int | bool],
    logprobs: Optional[int] = None,
    ignore_eos: bool = False,
    lora_modules: Optional[Iterable[str]] = None,
) -> List[RequestFuncOutput]:
    """Fire off *all* requests concurrently and collect streaming stats."""

    request_func = ASYNC_REQUEST_FUNCS[backend]

    tasks = []
    lora_iter = iter(lora_modules) if lora_modules else None

    for req in input_requests:
        req_model_id, req_model_name = model_id, model_name
        if lora_iter:
            req_lora_module = next(lora_iter)
            req_model_id, req_model_name = req_lora_module, req_lora_module

        body = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=req.prompt,
            api_url=api_url,
            prompt_len=req.prompt_len,
            output_len=req.expected_output_len,
            logprobs=logprobs,
            multi_modal_content=req.multi_modal_data,
            ignore_eos=ignore_eos,
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
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code
    )

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

    # Set default temperature if not specified (for compatibility)
    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0

    # Prepare LoRA modules if specified
    lora_modules = None
    if args.lora_modules:
        lora_modules = [
            random.choice(args.lora_modules)
            for _ in range(args.batch_size * args.measure_batches)
        ]

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
            logprobs=args.logprobs,
            ignore_eos=args.ignore_eos,
            lora_modules=lora_modules[:args.batch_size] if lora_modules else None,
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

    # For detailed results and comprehensive metrics
    all_batch_outputs = []
    all_requests = []

    # Check goodput configuration
    goodput_config_dict = check_goodput_args(args)
    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]
    selected_percentile_metrics = args.percentile_metrics.split(",")

    for batch_idx in range(args.measure_batches):
        st_batch = time.perf_counter()

        # Use appropriate LoRA modules for this batch
        batch_lora_modules = None
        if lora_modules:
            start_idx = batch_idx * args.batch_size
            end_idx = (batch_idx + 1) * args.batch_size
            batch_lora_modules = lora_modules[start_idx:end_idx]

        outputs = await _run_batch(
            backend=args.backend,
            api_url=api_url,
            model_id=args.model,
            model_name=args.served_model_name or args.model,
            tokenizer=tokenizer,
            input_requests=requests,
            sampling_params=sampling_params,
            logprobs=args.logprobs,
            ignore_eos=args.ignore_eos,
            lora_modules=batch_lora_modules,
        )
        total_wall_time += time.perf_counter() - st_batch
        total_success += sum(o.success for o in outputs)

        # Store outputs and requests for comprehensive metrics
        all_batch_outputs.extend(outputs)
        all_requests.extend(requests)

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

    # ------------------------------------------------------------------
    # Calculate metrics with full compatibility to benchmark_serving.py
    # ------------------------------------------------------------------
    # Collect all actual output lengths
    actual_output_lens = []
    all_ttfts = []
    all_e2els = []  # end-to-end latencies
    total_input_tokens = 0
    good_completed = 0

    # Calculate comprehensive metrics using collected outputs
    for i, output in enumerate(all_batch_outputs):
        if output.success:
            output_len = output.output_tokens or len(output.itl)
            actual_output_lens.append(output_len)
            total_input_tokens += args.input_len  # Use consistent input length

            # TTFT calculation (first token time) - use 0 for decode-only since no separate prefill
            ttft = 0.0 if not output.itl else output.itl[0] if len(output.itl) > 0 else 0.0
            all_ttfts.append(ttft)

            # E2EL calculation (total latency)
            e2el = output.latency if hasattr(output, 'latency') and output.latency else sum(output.itl)
            all_e2els.append(e2el)
        else:
            actual_output_lens.append(0)
            all_ttfts.append(0.0)
            all_e2els.append(0.0)

    # Calculate goodput if configured
    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(all_ttfts)
            slo_values.append(goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in goodput_config_dict:
            # Use measured_tpots (pad with zeros for failed requests)
            all_tpots_for_goodput = []
            success_idx = 0
            for output in all_batch_outputs:
                if output.success and success_idx < len(measured_tpots):
                    all_tpots_for_goodput.append(measured_tpots[success_idx])
                    success_idx += 1
                else:
                    all_tpots_for_goodput.append(0.0)
            valid_metrics.append(all_tpots_for_goodput)
            slo_values.append(goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(all_e2els)
            slo_values.append(goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    # Calculate comprehensive statistics
    itls_ms = np.array(measured_itls) * MILLISECONDS
    tpot_ms = np.array(measured_tpots) * MILLISECONDS
    ttft_ms = np.array(all_ttfts) * MILLISECONDS
    e2el_ms = np.array(all_e2els) * MILLISECONDS

    # Base statistics
    stats = {
        "mean_itl_ms": float(np.mean(itls_ms)) if len(itls_ms) > 0 else 0.0,
        "median_itl_ms": float(np.median(itls_ms)) if len(itls_ms) > 0 else 0.0,
        "p99_itl_ms": float(np.percentile(itls_ms, 99)) if len(itls_ms) > 0 else 0.0,
        "std_itl_ms": float(np.std(itls_ms)) if len(itls_ms) > 0 else 0.0,
        "mean_tpot_ms": float(np.mean(tpot_ms)) if len(tpot_ms) > 0 else 0.0,
        "median_tpot_ms": float(np.median(tpot_ms)) if len(tpot_ms) > 0 else 0.0,
        "std_tpot_ms": float(np.std(tpot_ms)) if len(tpot_ms) > 0 else 0.0,
        "mean_ttft_ms": float(np.mean(ttft_ms)) if len(ttft_ms) > 0 else 0.0,
        "median_ttft_ms": float(np.median(ttft_ms)) if len(ttft_ms) > 0 else 0.0,
        "std_ttft_ms": float(np.std(ttft_ms)) if len(ttft_ms) > 0 else 0.0,
        "mean_e2el_ms": float(np.mean(e2el_ms)) if len(e2el_ms) > 0 else 0.0,
        "median_e2el_ms": float(np.median(e2el_ms)) if len(e2el_ms) > 0 else 0.0,
        "std_e2el_ms": float(np.std(e2el_ms)) if len(e2el_ms) > 0 else 0.0,
    }

    # Add percentile statistics
    def add_percentiles(metric_name: str, data_ms: np.ndarray):
        if metric_name in selected_percentile_metrics and len(data_ms) > 0:
            for p in selected_percentiles:
                p_word = str(int(p)) if int(p) == p else str(p)
                stats[f"p{p_word}_{metric_name}_ms"] = float(np.percentile(data_ms, p))

    add_percentiles("ttft", ttft_ms)
    add_percentiles("tpot", tpot_ms)
    add_percentiles("itl", itls_ms)
    add_percentiles("e2el", e2el_ms)

    if measured_iteration_latencies:
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

    # Print metrics based on selected_percentile_metrics
    def process_one_metric(metric_attribute_name: str, metric_name: str, metric_header: str):
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(f"Mean {metric_name} (ms)           : {stats[f'mean_{metric_attribute_name}_ms']:.2f}")
        print(f"Median {metric_name} (ms)         : {stats[f'median_{metric_attribute_name}_ms']:.2f}")
        if f'std_{metric_attribute_name}_ms' in stats:
            print(f"Std-dev {metric_name} (ms)       : {stats[f'std_{metric_attribute_name}_ms']:.2f}")
        for p in selected_percentiles:
            p_word = str(int(p)) if int(p) == p else str(p)
            if f"p{p_word}_{metric_attribute_name}_ms" in stats:
                print(f"P{p_word} {metric_name} (ms)             : {stats[f'p{p_word}_{metric_attribute_name}_ms']:.2f}")

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    if 'mean_batch_iter_ms' in stats:
        print("-" * 60)
        print(f"Mean batch iter (ms)    : {stats['mean_batch_iter_ms']:.2f}")
        print(f"Median batch iter (ms)  : {stats['median_batch_iter_ms']:.2f}")
        print(f"P99 batch iter (ms)     : {stats['p99_batch_iter_ms']:.2f}")
        print(f"Std-dev batch iter (ms) : {stats['std_batch_iter_ms']:.2f}")
        tokens_per_sec = args.batch_size * 1000.0 / stats['mean_batch_iter_ms']
        print("-")
        print(f"Token throughput (tok/s): {tokens_per_sec:.1f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Return comprehensive results for saving (compatible with benchmark_serving.py)
    # ------------------------------------------------------------------
    total_output_tokens = sum(actual_output_lens)
    num_prompts = args.num_prompts or (args.batch_size * args.measure_batches)

    # Calculate throughput metrics
    request_throughput = total_success / total_wall_time if total_wall_time > 0 else 0.0
    request_goodput = good_completed / total_wall_time if total_wall_time > 0 else 0.0
    output_throughput = total_output_tokens / total_wall_time if total_wall_time > 0 else 0.0
    total_token_throughput = (total_input_tokens + total_output_tokens) / total_wall_time if total_wall_time > 0 else 0.0

    result = {
        # Basic execution info (compatible with benchmark_serving.py)
        "duration": total_wall_time,
        "completed": total_success,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": request_throughput,
        "request_goodput:": request_goodput if goodput_config_dict else None,  # Note the colon to match benchmark_serving.py
        "output_throughput": output_throughput,
        "total_token_throughput": total_token_throughput,

        # Decode-only specific fields
        "total_requests": args.measure_batches * len(requests),
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "skip_initial_tokens": args.skip_initial_tokens,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,

        # All statistics (includes ttft, tpot, itl, e2el, and percentiles)
        **stats,

        # Raw data arrays (compatible with benchmark_serving.py format)
        "input_lens": [req.prompt_len for req in requests] * args.measure_batches,
        "output_lens": actual_output_lens,
        "ttfts": all_ttfts,
        "itls": [output.itl for output in all_batch_outputs if output.success],
        "generated_texts": [output.generated_text for output in all_batch_outputs if output.success],
        "errors": [output.error for output in all_batch_outputs if not output.success],
    }

    if measured_iteration_latencies:
        result["token_throughput"] = tokens_per_sec

    return result


def main():
    args = _parse_args()

    # Print args for debugging (like benchmark_serving.py)
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run the benchmark
    benchmark_result = asyncio.run(main_async(args))

    # Save config and results to json if requested (compatible with benchmark_serving.py)
    if args.save_result or args.append_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = args.backend
        result_json["model_id"] = args.model
        tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts or (args.batch_size * args.measure_batches)

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic (compatible with benchmark_serving.py)
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        if not args.save_detailed:
            # Remove fields with too many data points (compatible with benchmark_serving.py)
            for field in [
                "input_lens",
                "output_lens",
                "ttfts",
                "itls",
                "generated_texts",
                "errors",
            ]:
                if field in result_json:
                    del result_json[field]

        # Save to file
        base_model_id = args.model.split("/")[-1]
        # Use compatible filename format with benchmark_serving.py
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}"
            if args.max_concurrency is not None
            else ""
        )
        file_name = f"{args.backend}-{args.batch_size}batch{max_concurrency_str}-{base_model_id}-{current_dt}.json"
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)

        with open(
            file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
        ) as outfile:
            # Append a newline if appending to existing file
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)

        save_to_pytorch_benchmark_format(args, result_json, file_name)
        print(f"Results saved to: {file_name}")


if __name__ == "__main__":
    main()