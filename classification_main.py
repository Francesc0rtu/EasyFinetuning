from src import ShapleyScorerFactory, TrainerFactory
from dataclasses import dataclass


@dataclass
class Config:
    model_type: str = "default"
    dataset_path: str = "data/dataset.csv"
    checkpoint: str = "checkpoints/default"
    batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    output_shaply_path: str = "data/shapley_values.csv"


config = Config()


def run(config):
    trainer = TrainerFactory.create(
        config.model_type,
        dataset_path=config.dataset_path,
    )
    trainer.train(
        output_dir=config.checkpoint,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
    )

    scorer = ShapleyScorerFactory.create(
        config.model_type,
        input_data_path=config.dataset_path,
        output_data_path=config.output_shaply_path,
        checkpoint=config.checkpoint,
    )
    label = "LABEL_1" if config.model_type == "default" else 1
    scorer.run_shap(
        label_value=label,
    )
    scorer.shap_values_for_words()
    scorer.save_shap_values()


if __name__ == "__main__":
    run(config)
