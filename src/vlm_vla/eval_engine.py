"""Evaluation engine for VLA models on LIBERO."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class TaskResult:
    task_name: str
    success_rate: float
    num_episodes: int
    avg_steps: float
    failures: dict  # {"miss": n, "drop": n, "timeout": n, "collision": n}


@dataclass
class EvalReport:
    model_name: str
    task_suite: str
    results: list  # list[TaskResult]

    @property
    def avg_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.success_rate for r in self.results) / len(self.results)

    def to_table(self) -> str:
        header = f"{'Task':<35} | {'Success':>8} | {'Episodes':>8} | {'Avg Steps':>9}"
        sep = "-" * len(header)
        rows = [header, sep]
        for r in self.results:
            rows.append(
                f"{r.task_name:<35} | {r.success_rate:>7.1%} | {r.num_episodes:>8} | {r.avg_steps:>9.1f}"
            )
        rows.append(sep)
        rows.append(
            f"{'Average':<35} | {self.avg_success_rate:>7.1%} | {'':>8} | {'':>9}"
        )
        return "\n".join(rows)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": self.model_name,
            "task_suite": self.task_suite,
            "avg_success_rate": self.avg_success_rate,
            "results": [
                {
                    "task_name": r.task_name,
                    "success_rate": r.success_rate,
                    "num_episodes": r.num_episodes,
                    "avg_steps": r.avg_steps,
                    "failures": r.failures,
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def compare_reports(*reports: EvalReport) -> str:
    """Generate comparison table across multiple models."""
    if not reports:
        return "No reports to compare."

    # Collect all task names
    all_tasks = []
    for r in reports:
        for tr in r.results:
            if tr.task_name not in all_tasks:
                all_tasks.append(tr.task_name)

    # Build header
    model_names = [r.model_name for r in reports]
    header = f"{'Task':<35}"
    for name in model_names:
        header += f" | {name:>15}"
    sep = "-" * len(header)

    rows = [header, sep]
    for task in all_tasks:
        row = f"{task:<35}"
        for report in reports:
            tr = next((r for r in report.results if r.task_name == task), None)
            val = f"{tr.success_rate:.1%}" if tr else "N/A"
            row += f" | {val:>15}"
        rows.append(row)

    rows.append(sep)
    avg_row = f"{'Average':<35}"
    for report in reports:
        avg_row += f" | {report.avg_success_rate:>14.1%}"
    rows.append(avg_row)

    return "\n".join(rows)
