import wandb


def safe_wandb_log(data: dict) -> None:
    """Log to wandb only if a run has been initialized.

    wandb.log() raises if called before wandb.init() — this lets call sites
    log unconditionally while still respecting use_wandb=false, since
    wandb.init() is only called when that flag is set.
    """
    if wandb.run is not None:
        wandb.log(data)
