"""
Unit Tests for Data Loader, Synthetic Telemetry Generator, and Prompt Formatting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.data_loader import (
    INSTRUCTION_TEXT,
    format_chatml_prompt,
    generate_synthetic_telemetry,
    load_raw_dataset,
)


SAMPLE_DATA_PATH = Path("data/sample_logs.jsonl")


def test_sample_logs_file_exists() -> None:
    """Ensure data/sample_logs.jsonl is present and non-empty."""
    assert SAMPLE_DATA_PATH.is_file(), f"Expected {SAMPLE_DATA_PATH} to exist."
    assert SAMPLE_DATA_PATH.stat().st_size > 0, "Dataset file must not be empty."


def test_sample_logs_schema() -> None:
    """Validate that every line in sample_logs.jsonl conforms to the required schema."""
    records = load_raw_dataset(str(SAMPLE_DATA_PATH))
    assert len(records) >= 100, f"Expected at least 100 records, got {len(records)}."

    for idx, row in enumerate(records):
        assert "instruction" in row, f"Row {idx} missing 'instruction'."
        assert "input" in row, f"Row {idx} missing 'input'."
        assert "output" in row, f"Row {idx} missing 'output'."

        assert isinstance(row["instruction"], str) and row["instruction"].strip(), f"Row {idx} invalid instruction."
        assert isinstance(row["input"], str) and row["input"].strip(), f"Row {idx} invalid input."
        assert isinstance(row["output"], str) and row["output"].strip(), f"Row {idx} invalid output."

        assert row["instruction"] == INSTRUCTION_TEXT


def test_sample_logs_structured_json_output() -> None:
    """Verify that the 'output' field of every record parses to valid JSON with required keys."""
    records = load_raw_dataset(str(SAMPLE_DATA_PATH))

    allowed_severities = {"CRITICAL", "HIGH", "MEDIUM"}

    for idx, row in enumerate(records):
        try:
            parsed = json.loads(row["output"])
        except json.JSONDecodeError as exc:
            pytest.fail(f"Row {idx} 'output' is not valid JSON: {exc}")

        assert "root_cause" in parsed, f"Row {idx} missing 'root_cause'."
        assert "severity" in parsed, f"Row {idx} missing 'severity'."
        assert "remediation" in parsed, f"Row {idx} missing 'remediation'."

        assert isinstance(parsed["root_cause"], str) and len(parsed["root_cause"]) > 5
        assert isinstance(parsed["remediation"], str) and len(parsed["remediation"]) > 5
        assert parsed["severity"] in allowed_severities, f"Invalid severity '{parsed['severity']}' at row {idx}."


def test_synthetic_generator_creates_valid_dataset(tmp_path: Path) -> None:
    """Verify the synthetic telemetry generator function creates a valid JSONL file."""
    temp_file = tmp_path / "temp_telemetry.jsonl"
    num_samples = 24

    generated = generate_synthetic_telemetry(num_samples=num_samples, output_path=str(temp_file))

    assert len(generated) == num_samples
    assert temp_file.is_file()

    # Read back and verify
    loaded = load_raw_dataset(str(temp_file))
    assert len(loaded) == num_samples

    for item in loaded:
        assert item["instruction"] == INSTRUCTION_TEXT
        assert len(item["input"]) > 20
        out_json = json.loads(item["output"])
        assert out_json["severity"] in {"CRITICAL", "HIGH", "MEDIUM"}


def test_format_chatml_prompt_structure() -> None:
    """Test ChatML prompt formatting for training and inference."""
    instruction = "Analyze the error."
    error_trace = "kernel: Out of memory"
    target_output = '{"root_cause": "OOM", "severity": "CRITICAL", "remediation": "Scale RAM"}'

    # Prompt with target output (training format)
    full_prompt = format_chatml_prompt(instruction, error_trace, target_output)
    assert "<|im_start|>system" in full_prompt
    assert "<|im_start|>user" in full_prompt
    assert "<|im_start|>assistant" in full_prompt
    assert "<|im_end|>" in full_prompt
    assert target_output in full_prompt

    # Prompt without target output (inference format)
    infer_prompt = format_chatml_prompt(instruction, error_trace, output_text=None)
    assert infer_prompt.endswith("<|im_start|>assistant\n")
    assert target_output not in infer_prompt


def test_tokenization_length_bounds() -> None:
    """Validate that typical formatted prompts stay well within max_length=512 limit."""
    records = load_raw_dataset(str(SAMPLE_DATA_PATH))
    max_estimated_tokens = 0

    for row in records:
        prompt = format_chatml_prompt(row["instruction"], row["input"], row["output"])
        # Standard conservative heuristic: 1 word ~ 1.3 tokens
        estimated_tokens = int(len(prompt.split()) * 1.3)
        if estimated_tokens > max_estimated_tokens:
            max_estimated_tokens = estimated_tokens

    # Verify all samples easily fit within 512 max length
    assert max_estimated_tokens < 512, f"Sample exceeded max_length token estimate: {max_estimated_tokens}"

