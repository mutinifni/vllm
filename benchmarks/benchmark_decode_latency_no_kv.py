#!/usr/bin/env python3
import argparse
import asyncio
import aiohttp
import json
import os
import time
import random
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm
import importlib.util
import sys

# Dynamically import metrics code from benchmark_serving.py
serving_path = os.path.join(os.path.dirname(__file__), "benchmark_serving.py")
spec = importlib.util.spec_from_file_location("benchmark_serving", serving_path)
serving = importlib.util.module_from_spec(spec)
sys.modules["benchmark_serving"] = serving
spec.loader.exec_module(serving)
BenchmarkMetrics = serving.BenchmarkMetrics
calculate_metrics = serving.calculate_metrics
check_goodput_args = serving.check_goodput_args
save_to_pytorch_benchmark_format = serving.save_to_pytorch_benchmark_format

def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark the online disaggregated decode-only vLLM server with KV cache load time exclusion."
    )
    parser.add_argument("--server-url", type=str, required=True, help="Base URL of the decode server (e.g., http://localhost:8000)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file from prefill phase.")
    parser.add_argument("--model", type=str, required=True, help="Model name (for API request body and tokenizer).")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name or path (defaults to model path).")
    parser.add_argument("--served-model-name", type=str, default=None, help="Served model name for API requests.")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Maximum number of concurrent requests.")
    parser.add_argument("--num-requests", type=int, default=None, help="Number of decode requests to send (default: all prompts in manifest).")
    parser.add_argument("--max-tokens", type=int, default=16, help="Number of tokens to generate in decode phase.")
    parser.add_argument("--logprobs", type=int, default=None, help="Number of logprobs-per-token to request.")
    parser.add_argument("--request-rate", type=float, default=float('inf'), help="Requests per second (QPS). Default: inf (send as fast as possible).")
    parser.add_argument("--burstiness", type=float, default=1.0, help="Burstiness factor for request timing. 1.0=Poisson, <1=bursty, >1=uniform.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--profile", action="store_true", help="Use Torch Profiler. The endpoint must be launched with VLLM_TORCH_PROFILER_DIR to enable profiler.")
    parser.add_argument("--save-result", action="store_true", help="Save benchmark results to a JSON file.")
    parser.add_argument("--save-detailed", action="store_true", help="Include per-request details in result file.")
    parser.add_argument("--append-result", action="store_true", help="Append results to existing file.")
    parser.add_argument("--metadata", metavar="KEY=VALUE", nargs="*", help="Key-value pairs for metadata in result JSON.")
    parser.add_argument("--result-dir", type=str, default=None, help="Directory to save result JSON.")
    parser.add_argument("--result-filename", type=str, default=None, help="Filename for saving results (default: timestamped).")
    parser.add_argument("--ignore-eos", action="store_true", help="Set ignore_eos flag in requests.")
    parser.add_argument("--percentile-metrics", type=str, default="ttft,tpot,itl", help="Comma-separated metrics for percentiles.")
    parser.add_argument("--metric-percentiles", type=str, default="99", help="Comma-separated percentiles (e.g., 50,90,99).")
    parser.add_argument("--goodput", nargs="+", required=False, help='Goodput SLOs as KEY:VALUE pairs in ms.')
    parser.add_argument("--stream", action="store_true", help="Use streaming requests (with usage included).")
    return parser

async def request_generator(prompts, request_rate, burstiness):
    """Yield (idx, prompt) at intervals determined by QPS and burstiness."""
    theta = 1.0 / (request_rate * burstiness) if request_rate != float('inf') else None
    for idx, prompt in enumerate(prompts):
        yield idx, prompt
        if request_rate == float('inf'):
            continue
        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(interval)

def extract_kv_load_time(chunk):
    """Extract KV cache load time from response chunk."""
    kv_transfer_params = chunk.get('kv_transfer_params', {})
    if kv_transfer_params:
        kv_load_time = kv_transfer_params.get('kv_load_time_ms')
        if kv_load_time is not None:
            return float(kv_load_time)
    return None

async def send_request(session, url, model, prompt, max_tokens, logprobs, ignore_eos, served_model_name, stream):
    data = {
        "model": served_model_name or model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.95,
        "stream": stream,
    }
    if stream:
        data["stream_options"] = {"include_usage": True}
    if logprobs is not None:
        data["logprobs"] = logprobs
    if ignore_eos:
        data["ignore_eos"] = True
    
    t0 = time.time()
    output_text = ""
    token_times = []
    kv_load_time_ms = None
    
    try:
        async with session.post(url, json=data) as resp:
            if resp.status == 200:
                if stream:
                    async for line in resp.content:
                        t = time.time()
                        if not line.strip():
                            continue
                        if not line.startswith(b"data: "):
                            continue
                        payload = line[len(b"data: "):].strip()
                        if payload == b"[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            
                            # Check for kv_transfer_params in streaming chunk
                            if 'kv_transfer_params' in chunk:
                                kv_time = extract_kv_load_time(chunk)
                                if kv_time is not None:
                                    kv_load_time_ms = kv_time
                            
                            if "choices" in chunk and chunk["choices"]:
                                choice = chunk["choices"][0]
                                if "delta" in choice:
                                    delta = choice["delta"]
                                    token = delta.get("content", "")
                                elif "text" in choice:
                                    token = choice["text"]
                                else:
                                    continue
                                
                                if token:
                                    output_text += token
                                    token_times.append(t)
                        except Exception:
                            continue
                else:
                    # Non-streaming request
                    response_data = await resp.json()
                    
                    # Extract kv_transfer_params from non-streaming response
                    if 'kv_transfer_params' in response_data:
                        kv_time = extract_kv_load_time(response_data)
                        if kv_time is not None:
                            kv_load_time_ms = kv_time
                    
                    if "choices" in response_data and response_data["choices"]:
                        output_text = response_data["choices"][0]["text"]
                        token_times = [time.time()]  # Single completion time
                
                latency = (token_times[-1] if token_times else time.time()) - t0
                if token_times:
                    ttft = token_times[0] - t0
                    itl = [t2 - t1 for t1, t2 in zip(token_times[:-1], token_times[1:])]
                else:
                    ttft = latency
                    itl = []
                return latency, output_text, None, ttft, itl, kv_load_time_ms
            else:
                err = f"ERROR: {resp.status} {await resp.text()}"
                latency = time.time() - t0
                return latency, None, err, latency, [], None
    except Exception as e:
        latency = time.time() - t0
        return latency, None, f"EXCEPTION: {str(e)}", latency, [], None

class AdjustedMetrics:
    """Extended metrics class that includes KV-adjusted values."""
    def __init__(self, original_metrics, kv_adjustments):
        # Copy all original metrics
        for key, value in original_metrics.__dict__.items():
            setattr(self, key, value)
        
        # Add KV-adjusted metrics
        self.kv_adjustments = kv_adjustments
        self.kv_hit_rate = len([x for x in kv_adjustments if x is not None]) / len(kv_adjustments) if kv_adjustments else 0
        self.mean_kv_load_time_ms = np.mean([x for x in kv_adjustments if x is not None]) if any(x is not None for x in kv_adjustments) else 0
        
        # Calculate adjusted metrics where KV load time is excluded
        if any(x is not None for x in kv_adjustments):
            # Adjust TTFT
            if hasattr(original_metrics, 'mean_ttft_ms'):
                adjusted_ttfts = []
                original_ttfts = getattr(original_metrics, '_ttft_values', [])
                for i, (ttft, kv_time) in enumerate(zip(original_ttfts, kv_adjustments)):
                    if kv_time is not None:
                        adjusted_ttfts.append(max(0, ttft - kv_time))
                    else:
                        adjusted_ttfts.append(ttft)
                
                if adjusted_ttfts:
                    self.mean_ttft_ms_adjusted = np.mean(adjusted_ttfts)
                    self.median_ttft_ms_adjusted = np.median(adjusted_ttfts)
                    self.std_ttft_ms_adjusted = np.std(adjusted_ttfts)
                    self.p99_ttft_ms_adjusted = np.percentile(adjusted_ttfts, 99)

async def main_async(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    with open(args.manifest, "r") as f:
        manifest = json.load(f)
    
    # Use all prompts in manifest if --num-requests is not specified
    if args.num_requests is None:
        args.num_requests = len(manifest)
    manifest = manifest[:args.num_requests]
    prompts = [entry["prompt"] for entry in manifest]
    
    url = args.server_url.rstrip("/") + "/v1/completions"
    connector = aiohttp.TCPConnector(limit=args.max_concurrency or 1000)
    sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    
    per_request = []
    kv_load_times = []
    pbar = None if args.disable_tqdm else tqdm(total=len(prompts))
    profile_started = False
    start_time = time.time()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        async def worker(idx, prompt):
            nonlocal profile_started
            if sem:
                async with sem:
                    latency, text, err, ttft, itl, kv_time = await send_request(
                        session, url, args.model, prompt, args.max_tokens, 
                        args.logprobs, args.ignore_eos, args.served_model_name, args.stream)
            else:
                latency, text, err, ttft, itl, kv_time = await send_request(
                    session, url, args.model, prompt, args.max_tokens, 
                    args.logprobs, args.ignore_eos, args.served_model_name, args.stream)
            
            per_request.append({
                "idx": idx,
                "latency": latency,
                "output": text,
                "prompt": prompt,
                "error": err,
                "ttft": ttft,
                "itl": itl,
                "kv_load_time_ms": kv_time,
            })
            kv_load_times.append(kv_time)
            
            if pbar is not None:
                pbar.update(1)
            elif not args.disable_tqdm:
                kv_str = f", kv_load={kv_time:.2f}ms" if kv_time is not None else ", no_kv"
                print(f"[KVExclusionBenchmark] Prompt {idx}: latency={latency:.3f}s{kv_str}, output={(text[:40] if text else err)!r}")
        
        # Profile support (start_profile)
        if args.profile and not profile_started:
            try:
                profile_url = url.replace("/v1/completions", "/start_profile")
                await session.post(profile_url)
                profile_started = True
            except Exception:
                pass
        
        tasks = []
        async for idx, prompt in request_generator(prompts, args.request_rate, args.burstiness):
            tasks.append(asyncio.create_task(worker(idx, prompt)))
        await asyncio.gather(*tasks)
        
        if pbar is not None:
            pbar.close()
        
        # Profile support (stop_profile)
        if args.profile and profile_started:
            try:
                profile_url = url.replace("/v1/completions", "/stop_profile")
                await session.post(profile_url)
            except Exception:
                pass
    
    total_time = time.time() - start_time

    # Load tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    # Prepare data for calculate_metrics
    class DummyOutput:
        def __init__(self, latency, prompt, output, error, ttft, itl):
            self.latency = latency
            self.prompt = prompt
            self.generated_text = output or ""
            self.error = error
            self.success = error is None
            self.prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            self.output_tokens = len(tokenizer(self.generated_text, add_special_tokens=False).input_ids)
            self.ttft = ttft
            self.itl = itl
    
    input_requests = []
    outputs_for_metrics = []
    ttft_values = []
    
    for req in per_request:
        input_requests.append(type('DummyRequest', (), {
            "prompt": req["prompt"], 
            "prompt_len": len(tokenizer(req["prompt"], add_special_tokens=False).input_ids), 
            "expected_output_len": len(tokenizer((req["output"] or ""), add_special_tokens=False).input_ids)
        })())
        outputs_for_metrics.append(DummyOutput(req["latency"], req["prompt"], req["output"], req["error"], req["ttft"], req["itl"]))
        ttft_values.append(req["ttft"] * 1000)  # Convert to ms

    # Metrics config
    selected_percentile_metrics = args.percentile_metrics.split(",")
    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]
    goodput_config_dict = check_goodput_args(args)
    
    # Calculate original metrics
    original_metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs_for_metrics,
        dur_s=total_time,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )
    
    # Store TTFT values for adjustment calculation
    original_metrics._ttft_values = ttft_values
    
    # Create adjusted metrics
    metrics = AdjustedMetrics(original_metrics, kv_load_times)
    
    # Print results
    print("\n==== KV Cache Load Time Exclusion Benchmark Results ====")
    print(f"Successful requests: {metrics.completed}")
    print(f"Benchmark duration (s): {total_time:.2f}")
    print(f"Total input tokens: {metrics.total_input}")
    print(f"Total generated tokens: {metrics.total_output}")
    print(f"Request throughput (req/s): {metrics.request_throughput:.2f}")
    print(f"Output token throughput (tok/s): {metrics.output_throughput:.2f}")
    print(f"Total Token throughput (tok/s): {metrics.total_token_throughput:.2f}")
    
    # KV cache metrics
    kv_hits = len([x for x in kv_load_times if x is not None])
    print(f"\n--- KV Cache Load Statistics ---")
    print(f"KV cache hits: {kv_hits}/{len(kv_load_times)} ({metrics.kv_hit_rate:.1%})")
    if kv_hits > 0:
        valid_kv_times = [x for x in kv_load_times if x is not None]
        print(f"Mean KV load time (ms): {np.mean(valid_kv_times):.2f}")
        print(f"Median KV load time (ms): {np.median(valid_kv_times):.2f}")
        print(f"Std KV load time (ms): {np.std(valid_kv_times):.2f}")
        print(f"Min KV load time (ms): {np.min(valid_kv_times):.2f}")
        print(f"Max KV load time (ms): {np.max(valid_kv_times):.2f}")
    
    # Original vs Adjusted metrics
    for metric_name in selected_percentile_metrics:
        print(f"\n--- {metric_name.upper()} Metrics ---")
        if hasattr(metrics, f"mean_{metric_name}_ms"):
            original_mean = getattr(metrics, f"mean_{metric_name}_ms")
            print(f"Original Mean {metric_name.upper()} (ms): {original_mean:.2f}")
            if hasattr(metrics, f"mean_{metric_name}_ms_adjusted"):
                adjusted_mean = getattr(metrics, f"mean_{metric_name}_ms_adjusted")
                improvement = ((original_mean - adjusted_mean) / original_mean) * 100 if original_mean > 0 else 0
                print(f"Adjusted Mean {metric_name.upper()} (ms): {adjusted_mean:.2f} (improvement: {improvement:.1f}%)")
        
        if hasattr(metrics, f"median_{metric_name}_ms"):
            original_median = getattr(metrics, f"median_{metric_name}_ms")
            print(f"Original Median {metric_name.upper()} (ms): {original_median:.2f}")
            if hasattr(metrics, f"median_{metric_name}_ms_adjusted"):
                adjusted_median = getattr(metrics, f"median_{metric_name}_ms_adjusted")
                print(f"Adjusted Median {metric_name.upper()} (ms): {adjusted_median:.2f}")
        
        if hasattr(metrics, f"std_{metric_name}_ms"):
            print(f"Std {metric_name.upper()} (ms): {getattr(metrics, f'std_{metric_name}_ms'):.2f}")
        
        if hasattr(metrics, f"percentiles_{metric_name}_ms"):
            for p, value in getattr(metrics, f"percentiles_{metric_name}_ms"):
                p_word = str(int(p)) if int(p) == p else str(p)
                print(f"P{p_word} {metric_name.upper()} (ms): {value:.2f}")
    
    print("========================================================\n")
    
    # Save results if requested
    if args.save_result or args.append_result:
        result_json = {}
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["model_id"] = args.model
        result_json["num_prompts"] = len(prompts)
        result_json["streaming"] = args.stream
        
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError("Invalid metadata format. Please use KEY=VALUE format.")
        
        result_json["request_rate"] = args.request_rate if args.request_rate < float("inf") else "inf"
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency
        result_json = {**result_json, **metrics.__dict__}
        
        if args.save_detailed:
            result_json["per_request"] = per_request
        
        base_model_id = args.model.split("/")[-1]
        max_concurrency_str = f"-concurrency{args.max_concurrency}" if args.max_concurrency is not None else ""
        stream_str = "-streaming" if args.stream else "-nonstreaming"
        file_name = f"kvexclusion-{args.request_rate}qps{max_concurrency_str}{stream_str}-{base_model_id}-{current_dt}.json"
        
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
        
        # Ensure all expected keys are present for save_to_pytorch_benchmark_format
        for k in [
            "p99_ttft_ms", "p99_tpot_ms", "p99_itl_ms",
            "median_ttft_ms", "mean_ttft_ms", "std_ttft_ms",
            "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms",
            "mean_itl_ms", "median_itl_ms", "std_itl_ms"
        ]:
            if k not in result_json:
                result_json[k] = 0.0
        
        with open(file_name, mode="a+" if args.append_result else "w", encoding="utf-8") as outfile:
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)
        
        save_to_pytorch_benchmark_format(args, result_json, file_name)
        print(f"Saved results to {file_name}")

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
