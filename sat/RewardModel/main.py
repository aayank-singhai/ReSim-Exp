from lightning.pytorch.cli import LightningCLI

from reward.dataset import LightningCARLA
from reward.model import RewardModel

cli = LightningCLI(
    RewardModel,
    LightningCARLA,
    seed_everything_default=32
)
