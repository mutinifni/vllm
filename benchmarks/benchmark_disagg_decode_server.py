#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys

# We reuse the full engine/server CLI from vLLM so that every flag supported
# by `vllm serve -h` is automatically accepted here as-is.
from vllm.engine.arg_utils import EngineArgs  # Provides add_cli_args
# Use vLLM's FlexibleArgumentParser which can handle deprecated kwargs
from vllm.utils import FlexibleArgumentParser


def _strip_kv_cache_args(argv: list[str]) -> list[str]:
    """Return a copy of *argv* without the --kv-cache-dir flag/value."""
    stripped: list[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ["--kv-cache-dir"]:
            skip_next = True  # Skip its value, too.
            continue
        if tok.startswith("--kv-cache-dir="):
            continue
        stripped.append(tok)
    return stripped


def main() -> None:
    # ------------------------------------------------------------------
    # Build CLI parser: all EngineArgs (same as `vllm serve`) + our flag.
    # ------------------------------------------------------------------
    parser = EngineArgs.add_cli_args(
        FlexibleArgumentParser(
            description=(
                "Start a decode-only vLLM OpenAI server that reuses KV "
                "caches generated in the pre-fill phase. All engine/server "
                "flags from `vllm serve -h` are accepted here."
            )
        )
    )

    # Script-specific arguments (not understood by `vllm serve`).
    parser.add_argument(
        "--kv-cache-dir",
        required=True,
        help="Path to the local_storage directory generated during prefill.",
    )

    # Parse known args so we can build the kv-transfer string; leave unknown
    # ones untouched (they will be forwarded verbatim to the server).
    args, unknown_cli = parser.parse_known_args()

    # ------------------------------------------------------------------
    # Construct kv_transfer_config JSON string.
    # ------------------------------------------------------------------
    kv_transfer_config = json.dumps(
        {
            "kv_connector": "SharedStorageConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"shared_storage_path": args.kv_cache_dir},
        }
    )

    # ------------------------------------------------------------------
    # Build the command line for the server subprocess.
    # ------------------------------------------------------------------
    # Start with the module-style invocation so we don't rely on a console
    # script being on PATH.
    cmd: list[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]

    # Forward every argument the user provided *except* --kv-cache-dir.
    forwarded_args = _strip_kv_cache_args(sys.argv[1:])

    cmd.extend(forwarded_args)
    # Inject the kv-transfer-config flag so the server sees the cache.
    cmd.extend(["--kv-transfer-config", kv_transfer_config])

    # ------------------------------------------------------------------
    # Launch.
    # ------------------------------------------------------------------
    print("[DisaggDecodeServer] Starting decode server:\n  " + " ".join(cmd))
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("[DisaggDecodeServer] Shutting down server…")
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()