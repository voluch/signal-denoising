from models.time_series_trasformer import TimeSeriesTransformer
from trainer import Trainer  # Your Trainer class from earlier

if __name__ == "__main__":
    # Settings
    input_dim = 1
    sequence_length = 1000  # or whatever your dataset uses
    dataset_type = "gaussian"  # or "non_gaussian"
    random_state = 42
    batch_size = 32
    epochs = 50
    learning_rate = 1e-4
    wandb_project = "signal-denoising"

    # Initialize model
    model = TimeSeriesTransformer(input_dim=input_dim)

    # Initialize and run trainer
    trainer = Trainer(
        model=model,
        model_name="TimeSeriesTransformer",
        dataset_type=dataset_type,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        random_state=random_state,
        wandb_project=wandb_project,
    )

    trainer.train()