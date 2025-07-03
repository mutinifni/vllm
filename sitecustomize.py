import os
import time
from typing import List
import importlib, logging  # noqa: E402
import sys
import threading
import numpy as np  # noqa: E402
from fastapi import APIRouter, FastAPI, Query, Request  # noqa: E402

print("[sitecustomize] module imported in process", file=sys.stderr)

# Enable this profiler with environment variable
if os.getenv("VLLM_BATCH_ITER_PROF", "0") != "1":
    # Profiling not enabled; define a stub endpoint that returns 404.
    try:
        from fastapi import APIRouter  # noqa: E402
        from vllm.entrypoints.openai import api_server  # noqa: E402

        router: APIRouter = api_server.router  # type: ignore

        @router.get("/iteration_stats")  # type: ignore
        async def iteration_stats_disabled():
            return {"error": "batch iteration profiler disabled"}
    except Exception:
        pass
    # Nothing else to do
else:
    # ---------------------------------------------------------
    # Patch LLMEngine.step to record batch-iteration latencies
    # ---------------------------------------------------------
    from vllm.engine.llm_engine import LLMEngine  # noqa: E402
    # Will attempt to import V1LLMEngine later lazily to avoid errors in old versions.

    _batch_iter_latencies: List[float] = []  # seconds - all latencies
    _step_classifications: List[str] = []  # "decode" or "prefill" for each step
    _original_step = LLMEngine.step  # type: ignore


    def _patched_step(self: LLMEngine, *args, **kwargs):  # type: ignore
        global _batch_iter_latencies, _step_classifications
        start_t = time.perf_counter()
        outputs = _original_step(self, *args, **kwargs)
        end_t = time.perf_counter()

        # Heuristic: count as a decode iteration iff no prefills scheduled.
        try:
            ctx = self.scheduler_contexts[0]
            sched_out = ctx.scheduler_outputs
            if sched_out is not None and sched_out.num_batched_tokens > 0:
                latency = end_t - start_t
                _batch_iter_latencies.append(latency)

                # Classify based on whether prefill groups were scheduled
                if sched_out.num_prefill_groups == 0:
                    _step_classifications.append("decode")
                else:
                    _step_classifications.append("prefill")
        except Exception:
            # Best-effort; never block serving path
            pass

        return outputs


    if not hasattr(LLMEngine, "_batch_iter_prof_patched"):
        LLMEngine.step = _patched_step  # type: ignore
        LLMEngine._batch_iter_prof_patched = True  # type: ignore

    # ---------------------------------------------------------
    # Register endpoint by patching FastAPI.__init__
    # ---------------------------------------------------------
    async def _iteration_stats_handler(request: Request):
        """Simple endpoint handler that returns all collected latencies"""
        global _batch_iter_latencies

        # Parse mode from query parameter
        mode = request.query_params.get("mode", "all")
        if mode not in ["all", "decode", "prefill"]:
            mode = "all"

        print(f"[sitecustomize] iteration_stats_handler called for mode: {mode}", file=sys.stderr)

        # Try to get latencies from worker processes via RPC
        worker_latencies = []
        try:
            engine_client = None

            # Method 1: Try to get engine_client via the request dependency
            try:
                api_server = importlib.import_module("vllm.entrypoints.openai.api_server")
                engine_client_func = getattr(api_server, "engine_client", None)
                if engine_client_func and callable(engine_client_func):
                    engine_client = engine_client_func(request)
                    print(f"[sitecustomize] Got engine_client: {type(engine_client)}", file=sys.stderr)
            except Exception as dep_err:
                print(f"[sitecustomize] Dependency method failed: {dep_err}", file=sys.stderr)

            # Method 2: Try to get from request.app.state if available
            if engine_client is None:
                try:
                    if hasattr(request, 'app') and hasattr(request.app, 'state'):
                        engine_client = getattr(request.app.state, 'engine_client', None)
                        print(f"[sitecustomize] Got engine_client from request.app.state: {type(engine_client)}", file=sys.stderr)
                except Exception as state_err:
                    print(f"[sitecustomize] Request.app.state method failed: {state_err}", file=sys.stderr)

            # Try the RPC call if we found an engine_client
            if engine_client is not None:
                rpc_method = f"get_{mode}_latencies"
                print(f"[sitecustomize] Calling RPC method: {rpc_method}", file=sys.stderr)

                if hasattr(engine_client, "collective_rpc_async"):
                    try:
                        results = await engine_client.collective_rpc_async(rpc_method)
                        print(f"[sitecustomize] RPC returned {len(results) if isinstance(results, list) else 0} items", file=sys.stderr)
                        if isinstance(results, list):
                            for item in results:
                                if isinstance(item, list):
                                    worker_latencies.extend(item)
                                elif isinstance(item, (int, float)):
                                    worker_latencies.append(item)
                    except Exception as rpc_err:
                        print(f"[sitecustomize] collective_rpc_async failed: {rpc_err}", file=sys.stderr)
                elif hasattr(engine_client, "collective_rpc"):
                    try:
                        results = await engine_client.collective_rpc(rpc_method)
                        print(f"[sitecustomize] RPC returned {len(results) if isinstance(results, list) else 0} items", file=sys.stderr)
                        if isinstance(results, list):
                            for item in results:
                                if isinstance(item, list):
                                    worker_latencies.extend(item)
                                elif isinstance(item, (int, float)):
                                    worker_latencies.append(item)
                    except Exception as rpc_err:
                        print(f"[sitecustomize] collective_rpc failed: {rpc_err}", file=sys.stderr)
                else:
                    print("[sitecustomize] No RPC methods found on engine_client", file=sys.stderr)
            else:
                print("[sitecustomize] No engine_client found", file=sys.stderr)

        except Exception as e:
            print(f"[sitecustomize] RPC setup failed: {e}", file=sys.stderr)

        # Combine local and worker latencies
        all_latencies = []
        if _batch_iter_latencies:
            all_latencies.extend([l * 1000.0 for l in _batch_iter_latencies])  # Convert to ms
        if worker_latencies:
            all_latencies.extend(worker_latencies)  # Assume already in ms from RPC

        print(f"[sitecustomize] Total latencies: {len(all_latencies)} (local: {len(_batch_iter_latencies)}, worker: {len(worker_latencies)})", file=sys.stderr)

        if not all_latencies:
            return {"count": 0, "latencies_ms": []}

        arr_ms = np.array(all_latencies)
        return {
            "count": len(arr_ms),
            "mean_ms": float(arr_ms.mean()),
            "median_ms": float(np.median(arr_ms)),
            "p99_ms": float(np.percentile(arr_ms, 99)),
            "std_ms": float(arr_ms.std()),
            "latencies_ms": arr_ms.tolist(),
        }

    # Monkey patch FastAPI.__init__ to add our endpoint
    _original_fastapi_init = FastAPI.__init__  # type: ignore
    def _fastapi_init_patch(self, *args, **kwargs):  # type: ignore
        _original_fastapi_init(self, *args, **kwargs)
        # Check if route already exists to avoid duplicates
        if not any(r.path == "/iteration_stats" for r in self.router.routes):
            try:
                self.add_api_route("/iteration_stats", _iteration_stats_handler, methods=["GET"])
                print("[sitecustomize] iteration_stats route added via FastAPI.__init__ patch", file=sys.stderr)
            except Exception as e:
                print(f"[sitecustomize] Failed to add route via FastAPI patch: {e}", file=sys.stderr)

    FastAPI.__init__ = _fastapi_init_patch  # type: ignore
    print("[sitecustomize] FastAPI.__init__ patched for endpoint registration", file=sys.stderr)

    # ---------------------------------------------------------
    # Patch V1LLMEngine.step to record batch-iteration latencies
    # ---------------------------------------------------------
    try:
        from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine  # noqa: E402
    except Exception as _v1_import_err:
        V1LLMEngine = None  # type: ignore
        logging.getLogger("sitecustomize").debug("Could not import V1 LLMEngine: %s", _v1_import_err)

    if V1LLMEngine is not None and not hasattr(V1LLMEngine, '_batch_iter_prof_patched'):
        _orig_v1_step = V1LLMEngine.step  # type: ignore
        def _v1_step_patch(self, *args, **kwargs):  # type: ignore
            global _batch_iter_latencies
            start = time.perf_counter()
            outputs = _orig_v1_step(self, *args, **kwargs)
            end = time.perf_counter()
            # V1 step always decode or prefill; but we can check scheduler_stats
            try:
                # outputs may be list of RequestOutput; self.engine_core has last scheduler_stats accessible
                if hasattr(self, 'engine_core'):
                    # engine_core.get_output sets scheduler_stats in its last output; not easily accessible.
                    pass
                # naive assumption: once past warmup, each step is decode; store regardless
                _batch_iter_latencies.append(end - start)
            except Exception:
                pass
            return outputs
        V1LLMEngine.step = _v1_step_patch  # type: ignore
        V1LLMEngine._batch_iter_prof_patched = True  # type: ignore

    # ---------------------------------------------------------
    # Patch EngineCore.step (V1) to record latencies and expose RPC util
    # ---------------------------------------------------------
    try:
        from vllm.v1.engine.core import EngineCore  # noqa: E402
    except Exception as _ec_import_err:
        EngineCore = None  # type: ignore
        logging.getLogger("sitecustomize").debug("Could not import EngineCore: %s", _ec_import_err)

    if EngineCore is not None and not hasattr(EngineCore, "_batch_iter_prof_patched"):
        print("[sitecustomize] Patching EngineCore.step for latency collection...", file=sys.stderr)
        _orig_ec_step = EngineCore.step  # type: ignore

        def _ec_step_patch(self, *args, **kwargs):  # type: ignore
            global _batch_iter_latencies, _step_classifications
            _start = time.perf_counter()
            outputs, executed = _orig_ec_step(self, *args, **kwargs)
            _end = time.perf_counter()
            # Record only if model executed something in this step.
            if executed:
                latency_s = _end - _start

                # Store in global list
                _batch_iter_latencies.append(latency_s)

                # Try to classify based on scheduler state if available
                step_type = "decode"  # Default assumption
                try:
                    # Try to access scheduler state to determine if this was prefill or decode
                    if hasattr(self, 'scheduler') and hasattr(self.scheduler, '_last_had_new_reqs'):
                        if self.scheduler._last_had_new_reqs:
                            step_type = "prefill"
                except:
                    # Fall back to heuristic: first few steps are likely prefill
                    if len(_batch_iter_latencies) <= 10:
                        step_type = "prefill"

                _step_classifications.append(step_type)

                print(f"[sitecustomize] EngineCore step executed ({step_type}), latency: {latency_s:.4f}s, total: {len(_batch_iter_latencies)}", file=sys.stderr)
            return outputs, executed

        EngineCore.step = _ec_step_patch  # type: ignore

        # Provide RPC getters
        def _get_decode_latencies(self):  # type: ignore
            global _batch_iter_latencies, _step_classifications
            # Return latencies for steps classified as decode
            decode_latencies = []
            for i, (latency, step_type) in enumerate(zip(_batch_iter_latencies, _step_classifications)):
                if step_type == "decode":
                    decode_latencies.append(latency * 1000.0)  # Convert to ms
            return decode_latencies

        def _get_prefill_latencies(self):  # type: ignore
            global _batch_iter_latencies, _step_classifications
            # Return latencies for steps classified as prefill
            prefill_latencies = []
            for i, (latency, step_type) in enumerate(zip(_batch_iter_latencies, _step_classifications)):
                if step_type == "prefill":
                    prefill_latencies.append(latency * 1000.0)  # Convert to ms
            return prefill_latencies

        def _get_all_latencies(self):  # type: ignore
            global _batch_iter_latencies
            # Return all latencies in milliseconds
            return [l * 1000.0 for l in _batch_iter_latencies]

        EngineCore.get_decode_latencies = _get_decode_latencies  # type: ignore
        EngineCore.get_prefill_latencies = _get_prefill_latencies  # type: ignore
        EngineCore.get_all_latencies = _get_all_latencies  # type: ignore
        EngineCore._batch_iter_prof_patched = True  # type: ignore
        print("[sitecustomize] EngineCore patches applied successfully", file=sys.stderr)

        # Patch collective_rpc to expose latencies via RPC
        _orig_ec_collective_rpc = EngineCore.collective_rpc  # type: ignore

        def _ec_collective_rpc_patch(self, method, timeout=None, args=(), kwargs=None):  # type: ignore
            if method == "get_decode_latencies":
                result = self.get_decode_latencies()  # type: ignore
            elif method == "get_prefill_latencies":
                result = self.get_prefill_latencies()  # type: ignore
            elif method == "get_all_latencies":
                result = self.get_all_latencies()  # type: ignore
            elif method == "get_batch_iter_latencies":
                result = self.get_all_latencies()  # type: ignore
            else:
                return _orig_ec_collective_rpc(self, method, timeout, args, kwargs)

            print(f"[sitecustomize] collective_rpc {method} returning {len(result)} items", file=sys.stderr)
            return result

        EngineCore.collective_rpc = _ec_collective_rpc_patch  # type: ignore
        print("[sitecustomize] EngineCore.collective_rpc patched", file=sys.stderr)

    # ---------------------------------------------------------
    # Note: Using the main _batch_iter_latencies list defined at top level
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Patch Scheduler.schedule to tag whether step had new requests (prefill)
    # ---------------------------------------------------------
    try:
        from vllm.v1.core.sched.scheduler import Scheduler  # noqa
    except Exception:
        Scheduler = None  # type: ignore

    if Scheduler is not None and not hasattr(Scheduler, "_batch_iter_prof_patched"):
        _orig_sched_schedule = Scheduler.schedule  # type: ignore

        def _sched_schedule_patch(self, *args, **kwargs):  # type: ignore
            out = _orig_sched_schedule(self, *args, **kwargs)
            # If any new requests scheduled ==> prefill work happened this step
            has_new = bool(getattr(out, "scheduled_new_reqs", []))
            self._last_had_new_reqs = has_new  # type: ignore
            return out

        Scheduler.schedule = _sched_schedule_patch  # type: ignore
        Scheduler._batch_iter_prof_patched = True  # type: ignore

    # ---------------------------------------------------------
    # Note: Simplified implementation - complex aggregation removed
    # The simple endpoint handler above accesses the global _batch_iter_latencies directly
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Note: Using simpler endpoint handler registered via _try_register above
    # The complex mode-based handler caused conflicts, so keeping it simple
    # ---------------------------------------------------------