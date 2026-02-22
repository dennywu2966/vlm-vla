"""Dataset utilities for VLA training."""
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def inspect_dataset(repo_id: str) -> dict:
    """Load and summarize a LeRobotDataset."""
    ds = LeRobotDataset(repo_id)
    sample = ds[0]
    info = {
        "repo_id": repo_id,
        "total_frames": len(ds),
        "num_episodes": ds.num_episodes,
        "features": {k: list(v.shape) for k, v in sample.items() if hasattr(v, "shape")},
        "fps": ds.fps,
    }
    return info


def get_task_descriptions(repo_id: str) -> list[str]:
    """Extract unique task descriptions from dataset."""
    ds = LeRobotDataset(repo_id)
    tasks = ds.meta.tasks
    return tasks
