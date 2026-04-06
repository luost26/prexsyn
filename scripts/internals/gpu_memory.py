from pathlib import Path

import click
import pandas as pd
import torch
from tqdm.auto import tqdm

from prexsyn.models import PrexSyn


def bytes_to_mib(value: int) -> float:
    return value / (1024**2)


def model_size_mib(model: torch.nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return bytes_to_mib(param_bytes + buffer_bytes)


def model_num_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def module_num_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def module_size_mib(module: torch.nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
    return bytes_to_mib(param_bytes + buffer_bytes)


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        scaled = value / 1_000_000_000
        suffix = "B"
    elif value >= 1_000_000:
        scaled = value / 1_000_000
        suffix = "M"
    elif value >= 1_000:
        scaled = value / 1_000
        suffix = "K"
    else:
        return str(value)

    if abs(scaled - round(scaled)) < 1e-9:
        return f"{int(round(scaled))}{suffix}"
    return f"{scaled:.1f}{suffix}"


def _loss_from_output(output):
    if torch.is_tensor(output):
        return output.float().sum()
    if isinstance(output, dict):
        values = [v.float().sum() for v in output.values() if torch.is_tensor(v)]
        if values:
            return torch.stack(values).sum()
    if isinstance(output, (list, tuple)):
        values = [v.float().sum() for v in output if torch.is_tensor(v)]
        if values:
            return torch.stack(values).sum()
    raise TypeError("Model output must contain at least one tensor to build a loss.")


@click.command()
@click.option(
    "-n",
    "--num-bb",
    "num_bbs",
    multiple=True,
    default=(50000, 100000, 200000, 500000),
    show_default=True,
    type=int,
    help="Maximum backbone index. Repeat option to benchmark multiple values.",
)
@click.option(
    "-b",
    "--batch-size",
    "batch_sizes",
    multiple=True,
    default=(1, 128, 1024),
    show_default=True,
    type=int,
    help="Batch size for inference. Repeat option to benchmark multiple values.",
)
@click.option(
    "-w",
    "--warmup",
    default=1,
    show_default=True,
    type=int,
    help="Number of warmup iterations before each measurement",
)
@click.option(
    "-o",
    "--output-csv",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional CSV output path for benchmark results",
)
def main(
    num_bbs: tuple[int, ...],
    batch_sizes: tuple[int, ...],
    warmup: int,
    output_csv: Path | None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    if warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if any(num_bb <= 0 for num_bb in num_bbs):
        raise ValueError("All --num-bb values must be > 0")
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("All --batch-size values must be > 0")

    all_rows = []
    total_runs = len(num_bbs) * len(batch_sizes)
    progress = tqdm(total=total_runs, desc="Benchmark runs", unit="run")

    for num_bb in num_bbs:
        model = PrexSyn(
            dim=1024,
            nhead=16,
            dim_feedforward=2048,
            num_layers=12,
            bb_embed_dim=1024,
            descriptor_configs={"ecfp4": {"descriptor_dim": 2048, "num_tokens": 4}},
            num_token_types=5,
            max_bb_index=num_bb - 1,
            max_rxn_index=115,
            pad_token=0,
            end_token=1,
            start_token=2,
            bb_token=3,
            rxn_token=4,
        ).cuda()

        for batch_size in batch_sizes:
            progress.set_postfix(num_bb=num_bb, batch_size=batch_size)
            inputs = {
                "descriptors": [("ecfp4", torch.zeros((batch_size, 2048), device="cuda"))],
                "token_types": torch.zeros((batch_size, 16), dtype=torch.long, device="cuda"),
                "bb_indices": torch.zeros((batch_size, 16), dtype=torch.long, device="cuda"),
                "rxn_indices": torch.zeros((batch_size, 16), dtype=torch.long, device="cuda"),
            }

            row = {
                "num_bb": num_bb,
                "batch_size": batch_size,
                "num_parameters": model_num_parameters(model),
                "model_size (MiB)": model_size_mib(model),
                "forward peak memory": None,
                "forward+backward peak memory": None,
                "bb_embedding num parameters": module_num_parameters(model.synthesis_embedder.bb_embedding),
                "bb_embedding size (MiB)": module_size_mib(model.synthesis_embedder.bb_embedding),
                "bb_head num parameters": module_num_parameters(model.synthesis_output.bb_head),
                "bb_head size (MiB)": module_size_mib(model.synthesis_output.bb_head),
            }

            # 1) Forward pass only
            model.eval()
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(**inputs)
            torch.cuda.synchronize()

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            with torch.no_grad():
                _ = model(**inputs)
            torch.cuda.synchronize()
            forward_peak_mib = bytes_to_mib(torch.cuda.max_memory_allocated())

            row["forward peak memory"] = forward_peak_mib

            # 2) Forward + backward pass
            model.train()
            model.zero_grad(set_to_none=True)
            for _ in range(warmup):
                model.zero_grad(set_to_none=True)
                warmup_output = model(**inputs)
                warmup_loss = _loss_from_output(warmup_output)
                warmup_loss.backward()
            torch.cuda.synchronize()

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            output = model(**inputs)
            loss = _loss_from_output(output)
            loss.backward()
            torch.cuda.synchronize()
            fw_bw_peak_mib = bytes_to_mib(torch.cuda.max_memory_allocated())

            row["forward+backward peak memory"] = fw_bw_peak_mib

            all_rows.append(row)
            progress.update(1)

        del model
        torch.cuda.empty_cache()

    progress.close()

    df = pd.DataFrame(all_rows)
    ordered_columns = [
        "num_bb",
        "batch_size",
        "num_parameters",
        "model_size (MiB)",
        "forward peak memory",
        "forward+backward peak memory",
        "bb_embedding num parameters",
        "bb_embedding size (MiB)",
        "bb_head num parameters",
        "bb_head size (MiB)",
    ]
    df = df[ordered_columns]

    count_columns = [
        "num_bb",
        "num_parameters",
        "bb_embedding num parameters",
        "bb_head num parameters",
    ]
    for col in count_columns:
        df[col] = df[col].map(format_count)

    print(df.to_string(index=False))

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, float_format="%.2f")
        print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
    main()
