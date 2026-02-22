"""Environment adapter for swappable simulation backends.

V1: LIBERO (MuJoCo) only.
V3+: Extend to Isaac Lab, real hardware.
"""
from dataclasses import dataclass


@dataclass
class ActionSpaceConfig:
    """Extensible action space definition.

    V1: 7D (6 DoF arm + gripper)
    V4: 30D+ (humanoid full body)
    """
    dims: int = 7
    groups: dict = None  # e.g. {"arm": 6, "gripper": 1}
    normalize: bool = True
    action_range: tuple = (-1.0, 1.0)

    def __post_init__(self):
        if self.groups is None:
            self.groups = {"arm": self.dims - 1, "gripper": 1}


# Default configs for supported environments
LIBERO_ACTION_SPACE = ActionSpaceConfig(
    dims=7,
    groups={"arm": 6, "gripper": 1},
)

# Future: Isaac Lab humanoid
# ISAAC_HUMANOID_ACTION_SPACE = ActionSpaceConfig(
#     dims=30,
#     groups={"left_arm": 7, "right_arm": 7, "torso": 3, "legs": 12, "gripper_l": 1, "gripper_r": 1},
# )
