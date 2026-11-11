"""
Data Loader and Synthetic Hardware Telemetry Dataset Generator.

Generates realistic multi-line system telemetry error traces (OOM, Kernel Panic,
Segfault, NVMe I/O timeouts, PCIe link drops, GPU ECC faults, etc.) and provides
tokenization pipelines for HuggingFace and PyTorch fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert systems reliability and hardware diagnostics AI engineer. "
    "Analyze the provided system telemetry trace, determine the exact root failure cause, "
    "assign appropriate severity (CRITICAL, HIGH, or MEDIUM), and recommend immediate remediation steps "
    "in strictly valid JSON format matching schema: "
    '{"root_cause": string, "severity": "CRITICAL"|"HIGH"|"MEDIUM", "remediation": string}.'
)

INSTRUCTION_TEXT = (
    "Analyze the system error trace, diagnose the root failure cause, and recommend immediate remediation."
)


def _generate_oom_sample(idx: int) -> Dict[str, str]:
    pid = random.randint(10000, 65535)
    total_vm_mb = random.randint(32768, 131072)
    rss_mb = int(total_vm_mb * random.uniform(0.85, 0.98))
    proc_name = random.choice(["python3", "ray::worker", "vllm_engine", "torchrun", "mysqld", "docker-containerd"])
    cgroup = random.choice(["/kubepods/burstable/pod12a8f", "/system.slice/ml-training.service", "/user.slice/job-992"])
    
    trace = (
        f"[Hardware Monitor] Node: edge-worker-0{random.randint(1, 9)} | Kernel: 5.15.0-89-generic\n"
        f"kernel: [{random.randint(10000, 99999)}.{random.randint(100, 999)}] Out of memory: Killed process {pid} ({proc_name}) "
        f"total-vm:{total_vm_mb * 1024}kB, anon-rss:{rss_mb * 1024}kB, file-rss:4096kB, shmem-rss:0kB\n"
        f"kernel: [{random.randint(10000, 99999)}.{random.randint(100, 999)}] memory: usage {total_vm_mb * 1024}kB, limit {total_vm_mb * 1024}kB, failcnt {random.randint(15, 200)}\n"
        f"kernel: [{random.randint(10000, 99999)}.{random.randint(100, 999)}] Memory cgroup out of memory: Killed process {pid} ({proc_name}) score {random.randint(800, 999)} in {cgroup}\n"
        f"systemd[1]: Process {pid} ({proc_name}) of unit ml-service.service failed with exit code 137 (SIGKILL)."
    )

    output = {
        "root_cause": f"Linux OOM Killer invoked against process '{proc_name}' (PID {pid}) after exceeding memory cgroup budget limit of {total_vm_mb // 1024}GB.",
        "severity": "CRITICAL",
        "remediation": f"Increase the memory limit in {cgroup}, implement dynamic batch sizing/gradient checkpointing to cap peak RSS under {rss_mb // 1024}GB, and enable swap or zram fallbacks.",
    }
    return {"instruction": INSTRUCTION_TEXT, "input": trace, "output": json.dumps(output)}


def _generate_kernel_panic_sample(idx: int) -> Dict[str, str]:
    cpu_id = random.randint(0, 31)
    rip_addr = f"0x{random.randint(0x100000, 0xFFFFFF):08x}"
    func = random.choice(["native_queued_spin_lock_slowpath", "handle_mm_fault", "do_page_fault", "kmem_cache_alloc"])
    
    trace = (
        f"[Syslog] System Exception Alert - Host: edge-gateway-x86-{random.randint(1, 5)}\n"
        f"kernel: [{random.randint(1000, 8999)}.{random.randint(10, 99)}] BUG: unable to handle page fault for address {rip_addr}\n"
        f"kernel: [{random.randint(1000, 8999)}.{random.randint(10, 99)}] #PF: supervisor read access in kernel mode\n"
        f"kernel: [{random.randint(1000, 8999)}.{random.randint(10, 99)}] #PF: error_code(0x0000) - not-present page\n"
        f"kernel: [{random.randint(1000, 8999)}.{random.randint(10, 99)}] CPU: {cpu_id} PID: {random.randint(100, 2000)} Comm: kworker/u64:{cpu_id} Tainted: G        W  OE 5.15.0-91-generic\n"
        f"kernel: [{random.randint(1000, 8999)}.{random.randint(10, 99)}] RIP: 0010:{func}+0x14/0x30\n"
        f"kernel: [{random.randint(1000, 8999)}.{random.randint(10, 99)}] Kernel panic - not syncing: Fatal exception in interrupt"
    )

    output = {
        "root_cause": f"Kernel panic triggered by invalid memory page fault (null/unmapped pointer dereference) in '{func}' executing on CPU {cpu_id}.",
        "severity": "CRITICAL",
        "remediation": "Reboot the host system immediately into rescue mode, inspect kernel crash dump via crash/kdump at the fault RIP address, and update out-of-tree kernel modules.",
    }
    return {"instruction": INSTRUCTION_TEXT, "input": trace, "output": json.dumps(output)}


def _generate_segfault_sample(idx: int) -> Dict[str, str]:
    lib = random.choice(["libtorch_cpu.so", "libonnxruntime.so.1.16", "libopenblas.so.0", "libcuda.so.1", "libggml.so"])
    app = random.choice(["telemetry_agent", "inference_server", "stream_processor", "quant_pipeline"])
    addr = f"{random.randint(0x10, 0x7FFF00):016x}"
    ip = f"{random.randint(0x7F000000, 0x7FFFFFFF):016x}"

    trace = (
        f"[OS Runtime Audit] Service Monitoring Daemon\n"
        f"dmesg: [{random.randint(2000, 9999)}.{random.randint(100, 999)}] {app}[{random.randint(10000, 40000)}]: "
        f"segfault at {addr} ip {ip} sp {random.randint(0x70000000, 0x7FFFFFFF):016x} error 4 in {lib}\n"
        f"systemd-coredump[8192]: Process {random.randint(10000, 40000)} ({app}) of user 1001 dumped core.\n"
        f"Core was generated by `/usr/local/bin/{app} --config /etc/edge/config.yaml`.\n"
        f"Program terminated with signal SIGSEGV, Segmentation fault."
    )

    output = {
        "root_cause": f"Userspace process '{app}' crashed with SIGSEGV caused by illegal memory access (error code 4, user-mode read) in shared library '{lib}'.",
        "severity": "HIGH",
        "remediation": f"Inspect core dump using 'gdb {app} core', check memory bounds around '{lib}' allocations, and recompile native bindings with AddressSanitizer (ASan).",
    }
    return {"instruction": INSTRUCTION_TEXT, "input": trace, "output": json.dumps(output)}


def _generate_nvme_io_sample(idx: int) -> Dict[str, str]:
    device = f"nvme{random.randint(0, 3)}n1"
    sector = random.randint(10000000, 900000000)
    timeout_sec = random.choice([30, 60, 120])

    trace = (
        f"[Storage Subsystem Telemetry] Controller NVMe Bus 0000:03:00.0\n"
        f"kernel: [{random.randint(5000, 40000)}.{random.randint(10, 99)}] nvme {device}: I/O {random.randint(100, 999)} QID {random.randint(1, 8)} timeout, aborting\n"
        f"kernel: [{random.randint(5000, 40000)}.{random.randint(10, 99)}] nvme {device}: I/O {random.randint(100, 999)} QID {random.randint(1, 8)} timeout, reset controller\n"
        f"kernel: [{random.randint(5000, 40000)}.{random.randint(10, 99)}] nvme {device}: Device not ready; aborting initialisation, error={random.choice([-19, -110])}\n"
        f"kernel: [{random.randint(5000, 40000)}.{random.randint(10, 99)}] blk_update_request: I/O error, dev {device}, sector {sector} op 0x1:(WRITE) flags 0x800 phys_seg 8 prio class 2\n"
        f"smartd[1044]: Device: /dev/{device}, 14 Currently unreadable (pending) sectors detected."
    )

    output = {
        "root_cause": f"Hardware I/O timeout and controller hang on NVMe SSD '{device}' leading to write failure at sector {sector} and unreadable pending sectors.",
        "severity": "CRITICAL",
        "remediation": f"Immediately cordon the host, unmount the filesystem on '/dev/{device}' to prevent corruption, trigger NVMe SMART health self-test, and replace the failing NVMe SSD drive.",
    }
    return {"instruction": INSTRUCTION_TEXT, "input": trace, "output": json.dumps(output)}


def _generate_pcie_aer_sample(idx: int) -> Dict[str, str]:
    bus_id = f"0000:{random.randint(1, 8):02x}:00.0"
    dev_name = random.choice(["NVIDIA RTX 4000 Ada", "Coral Edge TPU", "Intel I225-V Ethernet", "Hailo-8 AI Accelerator"])

    trace = (
        f"[PCIe Subsystem Diagnostic] Hardware Bus Topology Monitor\n"
        f"kernel: [{random.randint(3000, 25000)}.{random.randint(10, 99)}] pcieport 0000:00:01.0: AER: Uncorrected (Fatal) error received: {bus_id}\n"
        f"kernel: [{random.randint(3000, 25000)}.{random.randint(10, 99)}] {bus_id} PCIe Bus Error: severity=Uncorrected (Fatal), type=Transaction Layer, (Receiver Error)\n"
        f"kernel: [{random.randint(3000, 25000)}.{random.randint(10, 99)}] {bus_id}   device [{dev_name}]\n"
        f"kernel: [{random.randint(3000, 25000)}.{random.randint(10, 99)}] {bus_id}   [ 0] RxErr                  (First)\n"
        f"kernel: [{random.randint(3000, 25000)}.{random.randint(10, 99)}] {bus_id}   TLP Header: 04000001 0000000f 00000000 00000000\n"
        f"kernel: [{random.randint(3000, 25000)}.{random.randint(10, 99)}] pcieport 0000:00:01.0: PCIe Bus Error: severity=Fatal, device has fallen off the bus."
    )

    output = {
        "root_cause": f"PCIe Advanced Error Reporting (AER) fatal uncorrected physical/transaction receiver error on bus {bus_id} ({dev_name}), causing device disconnection.",
        "severity": "CRITICAL",
        "remediation": f"Power-cycle the edge appliance, inspect physical seating/power cables of '{dev_name}', clean PCIe gold contacts, or downgrade PCIe link speed from Gen4 to Gen3 in BIOS for signal stability.",
    }
    return {"instruction": INSTRUCTION_TEXT, "input": trace, "output": json.dumps(output)}


def _generate_gpu_ecc_sample(idx: int) -> Dict[str, str]:
    gpu_id = random.randint(0, 3)
    xid = random.choice([62, 79, 92, 45])
    temp = random.randint(92, 102)

    trace = (
        f"[NVIDIA SMI Event Log] Node: edge-ai-compute-0{random.randint(1, 4)}\n"
        f"NVRM: Xid (PCI:0000:0{gpu_id + 1}:00.0): {xid}, pid=18432, 0000 00000000 00000000 00000000 00000000\n"
        f"NVRM: GPU 0000:0{gpu_id + 1}:00.0: Uncorrectable double-bit ECC error detected in SRAM Bank {random.randint(0, 7)}!\n"
        f"nvidia-smi: GPU {gpu_id} Temp: {temp}C (Slowdown Temp: 93C, Shutdown Temp: 98C)\n"
        f"NVRM: GPU {gpu_id} thermal alert: Critical thermal threshold reached. Heavy SW thermal throttling active.\n"
        f"cuda-runtime: CUDA_ERROR_HARDWARE_STACK_ERROR (error 719) in cuLaunchKernel."
    )

    output = {
        "root_cause": f"NVIDIA GPU {gpu_id} uncorrectable double-bit SRAM ECC error (Xid {xid}) compounded by severe thermal throttling at {temp}C.",
        "severity": "CRITICAL",
        "remediation": f"Drain workload from GPU {gpu_id}, inspect cooling fan integrity and thermal paste, check chassis airflow, and replace the GPU board if SRAM ECC persists.",
    }
    return {"instruction": INSTRUCTION_TEXT, "input": trace, "output": json.dumps(output)}


def _generate_cpu_lockup_sample(idx: int) -> Dict[str, str]:
    cpu = random.randint(0, 15)
    stuck_time = random.randint(22, 60)
    process = random.choice(["kworker/u32:1", "quant_bench", "systemd-journal", "gzip"])

    trace = (
        f"[Kernel Watchdog Alert] Host: edge-node-{random.randint(10, 99)}\n"
        f"kernel: [{random.randint(8000, 30000)}.{random.randint(10, 99)}] watchdog: BUG: soft lockup - CPU#{cpu} stuck for {stuck_time}s! [{process}:{random.randint(1000, 9000)}]\n"
        f"kernel: [{random.randint(8000, 30000)}.{random.randint(10, 99)}] Modules linked in: vhost_net xt_conntrack ipt_MASQUERADE\n"
        f"kernel: [{random.randint(8000, 30000)}.{random.randint(10, 99)}] CPU: {cpu} PID: {random.randint(1000, 9000)} Comm: {process} Not tainted 5.15.0-88-generic\n"
        f"kernel: [{random.randint(8000, 30000)}.{random.randint(10, 99)}] RIP: 0010:smp_call_function_many_cond+0x110/0x2c0\n"
        f"kernel: [{random.randint(8000, 30000)}.{random.randint(10, 99)}] Call Trace:\n"
        f"kernel: [{random.randint(8000, 30000)}.{random.randint(10, 99)}]  <IRQ>\n"
        f"kernel: [{random.randint(8000, 30000)}.{random.randint(10, 99)}]  on_each_cpu_cond_mask+0x3c/0x60\n"
        f"kernel: [{random.randint(8000, 30000)}.{random.randint(10, 99)}]  flush_tlb_mm_range+0x85/0x100"
    )

    output = {
        "root_cause": f"Kernel soft lockup on CPU#{cpu} for {stuck_time}s during inter-processor TLB shootdown interrupt (IPI) flush in '{process}'.",
        "severity": "HIGH",
        "remediation": f"Tune sysctl 'kernel.watchdog_thresh=30', inspect CPU frequency scaling (cpufreq governor), and isolate CPU#{cpu} using taskset/cgroups for real-time threads.",
    }
    return {"instruction": INSTRUCTION_TEXT, "input": trace, "output": json.dumps(output)}


def _generate_nic_exhaustion_sample(idx: int) -> Dict[str, str]:
    interface = random.choice(["eth0", "enp3s0", "ens1f0", "wlan0"])
    drops = random.randint(10000, 500000)

    trace = (
        f"[Network Telemetry Collector] Device: {interface}\n"
        f"kernel: [{random.randint(12000, 45000)}.{random.randint(10, 99)}] {interface}: RX ring buffer overrun: {drops} packets dropped in last 60s\n"
        f"netlink: [QoS Alert] UDP socket buffer full: recv_buf_errors={drops // 2}\n"
        f"ethtool -S {interface}: rx_no_buffer_count: {drops}\n"
        f"ethtool -S {interface}: rx_missed_errors: {drops // 4}\n"
        f"systemd[1]: high packet loss detected on edge ingress telemetry port 8080."
    )

    output = {
        "root_cause": f"Network interface '{interface}' hardware RX ring buffer overflow causing packet drop rate of {drops} pkts/min due to kernel polling starvation.",
        "severity": "MEDIUM",
        "remediation": f"Increase NIC ring buffer descriptors via 'ethtool -G {interface} rx 4096', increase 'sysctl -w net.core.netdev_max_backlog=10000', and enable RPS/RFS.",
    }
    return {"instruction": INSTRUCTION_TEXT, "input": trace, "output": json.dumps(output)}


GENERATORS = [
    _generate_oom_sample,
    _generate_kernel_panic_sample,
    _generate_segfault_sample,
    _generate_nvme_io_sample,
    _generate_pcie_aer_sample,
    _generate_gpu_ecc_sample,
    _generate_cpu_lockup_sample,
    _generate_nic_exhaustion_sample,
]


def generate_synthetic_telemetry(num_samples: int = 100, output_path: str = "data/sample_logs.jsonl") -> List[Dict[str, str]]:
    """
    Generate diverse, realistic hardware and system error logs for instruction tuning.

    Args:
        num_samples: Total number of synthetic samples to generate.
        output_path: Path to save the output JSONL file.

    Returns:
        List of generated dictionary samples.
    """
    random.seed(42)
    samples: List[Dict[str, str]] = []

    for i in range(num_samples):
        generator_fn = GENERATORS[i % len(GENERATORS)]
        sample = generator_fn(i)
        samples.append(sample)

    # Shuffle to ensure even distribution across categories
    random.shuffle(samples)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Successfully generated %d synthetic telemetry logs at: %s", len(samples), out_file.resolve())
    return samples


def format_chatml_prompt(instruction: str, input_text: str, output_text: Optional[str] = None) -> str:
    """
    Format a single instruction-tuning sample into standard ChatML format.

    Args:
        instruction: Prompt instruction string.
        input_text: Raw system error trace.
        output_text: Ground truth assistant response (optional, omitted during inference).

    Returns:
        Formatted ChatML string.
    """
    user_content = f"{instruction}\n\n[SYSTEM ERROR TRACE]:\n{input_text.strip()}"
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if output_text is not None:
        prompt += f"{output_text.strip()}<|im_end|>\n"
    return prompt


def load_raw_dataset(data_path: str) -> List[Dict[str, str]]:
    """
    Load raw JSONL dataset from disk.

    Args:
        data_path: Path to the JSONL dataset file.

    Returns:
        List of loaded dictionaries.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry dataset file not found at: {path.resolve()}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to decode JSON at line %d: %s", line_no, exc)
                continue

            if "instruction" in data and "input" in data and "output" in data:
                records.append(data)
            else:
                logger.warning("Line %d missing required fields; skipping.", line_no)

    logger.info("Loaded %d valid telemetry records from %s", len(records), path)
    return records


def prepare_hf_dataset(
    data_path: str,
    tokenizer: Any = None,
    max_length: int = 512,
    train_ratio: float = 0.85,
    seed: int = 42,
) -> Tuple[Any, Any]:
    """
    Load JSONL dataset, format into ChatML prompts, and return train/eval splits.

    Supports Hugging Face `datasets.Dataset` when installed; falls back to structured lists.

    Args:
        data_path: Path to JSONL data.
        tokenizer: Optional Hugging Face Tokenizer instance.
        max_length: Token truncation / dynamic padding limit.
        train_ratio: Fraction of data allocated to training split.
        seed: Random seed for deterministic splitting.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    raw_records = load_raw_dataset(data_path)
    if not raw_records:
        raise ValueError(f"Dataset at {data_path} contains no valid records.")

    random.seed(seed)
    shuffled = list(raw_records)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_raw = shuffled[:split_idx]
    eval_raw = shuffled[split_idx:]

    def _format_records(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
        formatted = []
        for r in records:
            text = format_chatml_prompt(
                instruction=r["instruction"],
                input_text=r["input"],
                output_text=r["output"],
            )
            formatted.append({"text": text, "instruction": r["instruction"], "input": r["input"], "output": r["output"]})
        return formatted

    train_formatted = _format_records(train_raw)
    eval_formatted = _format_records(eval_raw)

    try:
        from datasets import Dataset

        train_ds = Dataset.from_list(train_formatted)
        eval_ds = Dataset.from_list(eval_formatted)

        if tokenizer is not None:
            def _tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
                tokenized = tokenizer(
                    batch["text"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None,
                )
                tokenized["labels"] = list(tokenized["input_ids"])
                return tokenized

            train_ds = train_ds.map(_tokenize_fn, batched=True, remove_columns=["text", "instruction", "input", "output"])
            eval_ds = eval_ds.map(_tokenize_fn, batched=True, remove_columns=["text", "instruction", "input", "output"])

        return train_ds, eval_ds

    except ImportError:
        logger.warning("Hugging Face 'datasets' library not installed. Returning raw split lists.")
        return train_formatted, eval_formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic hardware telemetry data and inspect format.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of telemetry samples to generate (default: 100).")
    parser.add_argument("--output", type=str, default="data/sample_logs.jsonl", help="Output file path.")
    args = parser.parse_args()

    samples = generate_synthetic_telemetry(num_samples=args.num_samples, output_path=args.output)
    print(f"Generated {len(samples)} samples. Inspecting sample 0:")
    print("--- CHATML PROMPT FORMAT ---")
    sample_chatml = format_chatml_prompt(
        instruction=samples[0]["instruction"],
        input_text=samples[0]["input"],
        output_text=samples[0]["output"],
    )
    print(sample_chatml[:600] + "\n...[TRUNCATED]")


if __name__ == "__main__":
    main()

