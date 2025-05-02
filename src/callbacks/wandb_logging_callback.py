from transformers import TrainerCallback
import wandb

class WandbLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        wandb.log({
            "current_step": state.global_step,
            "max_steps": state.max_steps,
            "current_epoch": state.epoch,
            "max_epochs": args.num_train_epochs,
        })

    def on_train_end(self, args, state, control, **kwargs):
        wandb.run.summary["final_step"] = state.global_step
        wandb.run.summary["final_epoch"] = state.epoch
        wandb.run.summary["early_stopped"] = (state.global_step < state.max_steps)
