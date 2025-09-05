"""Provides a detailed example of fine-tuning a TabPFNRegressor model.

This script demonstrates the complete workflow, including data loading and preparation
for the Bike Sharing Demand dataset, model configuration, the fine-tuning loop,
and performance evaluation for a regression task.

Note: We recommend running the fine-tuning scripts on a CUDA-enabled GPU, as full
support for the Apple Silicon (MPS) backend is still under development.
"""

from functools import partial
import os
import numpy as np
import sklearn.datasets
import torch
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor, TabPFNClassifier
from tabpfn_extensions.hpo import (
    TunedTabPFNRegressor,
)
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
import pandas as pd
import argparse

from tabpfn.model.encoders import SeqEncStep
from typing import Any, Literal, overload
from torch import nn


#########################################################################
class FeatureEmbedding(SeqEncStep):
    """A feature-wise encoder step."""

    def __init__(
        self,
        *,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("output",),
    ):
        super().__init__(in_keys, out_keys)
        self.layer = nn.Linear(1, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero
        self.list_layer = None

    def _fit(self, *x: torch.Tensor, **kwargs: Any):
        """Fit the encoder step. Make a list of # features of embedding."""
        new_x = torch.cat(x, dim=-1)
        l = self.layer
        self.list_layer = [l for _ in range(new_x.shape[-1])]

    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Apply the linear transformation to the each feature of the input.

        Args:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)  # type: ignore

        # Ensure input tensor dtype matches the layer's weight dtype
        # Since this layer gets input from the outside we verify the dtype
        x = x.to(self.list_layer[0].weight.dtype)

        # One embedding per feature
        for i in range(x.shape[-1]):
            if i == 0:
                w_base = self.list_layer[i](x[:, :, i].reshape(x.shape[0], x.shape[1], 1))
            else:
                w = self.list_layer[i](x[:, :, i].reshape(x.shape[0], x.shape[1], 1))
                w_base += w
        
        return (w_base,)
#########################################################################




def prepare_data(config: dict, id: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads, subsets, and splits the California Housing dataset."""
    print("--- 1. Data Preparation ---")
    if id == 0:
        # Fetch Ames housing data from OpenML
        bike_sharing = sklearn.datasets.fetch_openml(
            name="Bike_Sharing_Demand", version=2, 
            as_frame=True, parser="auto"
        )
    else:
        # Fetch Ames housing data from OpenML
        bike_sharing = sklearn.datasets.fetch_openml(
            # name="Bike_Sharing_Demand", version=2, 
            data_id=id,
            as_frame=True, parser="auto"
        )

    # Separate features (X) and target (y)
    X_df = bike_sharing.data
    y_df = bike_sharing.target

    # Select only numeric features for simplicity
    X_numeric = X_df.select_dtypes(include=np.number)

    X_all, y_all = X_numeric.values, y_df.values

    rng = np.random.default_rng(config["random_seed"])
    num_samples_to_use = min(config["num_samples_to_use"], len(y_all))
    indices = rng.choice(np.arange(len(y_all)), size=num_samples_to_use, replace=False)
    X, y = X_all[indices], y_all[indices]

    splitter = partial(
        train_test_split,
        test_size=config["valid_set_ratio"],
        random_state=config["random_seed"],
    )
    X_train, X_test, y_train, y_test = splitter(X, y)

    print(
        f"Loaded and split data: {X_train.shape[0]} train, {X_test.shape[0]} test samples."
    )
    print("---------------------------\n")
    return X_train, X_test, y_train, y_test


def setup_model_and_optimizer(config: dict) -> tuple[TabPFNClassifier, Optimizer, dict]:
    """Initializes the TabPFN classifier, optimizer, and training configs."""
    print("--- 2. Model and Optimizer Setup ---")
    classifier_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 2,
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
    }
    classifier = TabPFNClassifier(
        **classifier_config, fit_mode="batched", differentiable_input=False
    )
    classifier._initialize_model_variables()
    # Optimizer uses finetuning-specific learning rate
    optimizer = Adam(
        classifier.model_.parameters(), lr=config["finetuning"]["learning_rate"]
    )

    print(f"Using device: {config['device']}")
    print(f"Optimizer: Adam, Finetuning LR: {config['finetuning']['learning_rate']}")
    print("----------------------------------\n")
    return classifier, optimizer, classifier_config


def evaluate_model(
    classifier: TabPFNClassifier,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Evaluates the model's performance on the test set."""
    eval_classifier = clone_model_for_evaluation(
        classifier, eval_config, TabPFNClassifier
    )
    eval_classifier.fit(X_train, y_train)

    try:
        probabilities = eval_classifier.predict_proba(X_test)
        predictions = eval_classifier.predict(X_test)
        # print(f"probabilities.shape: {probabilities.shape} \n")
        # print(f"y_test.shape: {y_test.shape} \n")
        roc_auc = roc_auc_score(
            y_test, probabilities[:, 1], multi_class="ovr", average="weighted"
        )
        accuracy = accuracy_score(y_test, predictions)
        log_loss_score = log_loss(y_test, probabilities)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        roc_auc, log_loss_score, accuracy = np.nan, np.nan, np.nan

    return roc_auc, log_loss_score, accuracy


def main(id: int = 0, 
         FeatureEmbed: bool = False, 
         LR: float = 5e-6,
         EPOCHS: int = 30,
         SEED: int = 42) -> None:
    
    print(f"Running finetuning for dataset: {id} \n")
    

    """Main function to configure and run the finetuning workflow."""
    # --- Master Configuration ---
    # This improved structure separates general settings from finetuning hyperparameters.
    config = {
        # Sets the computation device ('cuda' for GPU if available, otherwise 'cpu').
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # The total number of samples to draw from the full dataset. This is useful for
        # managing memory and computation time, especially with large datasets.
        # For very large datasets the entire dataset is preprocessed and then
        # fit in memory, potentially leading to OOM errors.
        "num_samples_to_use": 100_000,
        # A seed for random number generators to ensure that data shuffling, splitting,
        # and model initializations are reproducible.
        "random_seed": SEED,
        # The proportion of the dataset to allocate to the valid set for final evaluation.
        "valid_set_ratio": 0.3,
        # During evaluation, this is the number of samples from the training set given to the
        # model as context before it makes predictions on the test set.
        "n_inference_context_samples": 10000,
    }
    config["finetuning"] = {
        # The total number of passes through the entire fine-tuning dataset.
        "epochs": EPOCHS,
        # A small learning rate is crucial for fine-tuning to avoid catastrophic forgetting.
        "learning_rate": LR,
        # Meta Batch size for finetuning, i.e. how many datasets per batch. Must be 1 currently.
        "meta_batch_size": 1,
        # The number of samples within each training data split. It's capped by
        # n_inference_context_samples to align with the evaluation setup.
        "batch_size": int(
            min(
                config["n_inference_context_samples"],
                config["num_samples_to_use"] * (1 - config["valid_set_ratio"]),
            )
        ),
    }


    # --- Setup Data, Model, and Dataloader ---
    # Use the modified data preparation function
    X_train, X_test, y_train, y_test = prepare_data(config, id)

    classifier, optimizer, classifier_config = setup_model_and_optimizer(config)

    
    splitter = partial(train_test_split, test_size=config["valid_set_ratio"])
    training_datasets = classifier.get_preprocessed_datasets(
        X_train, y_train, splitter, config["finetuning"]["batch_size"]
    )
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    )
    loss_function = torch.nn.CrossEntropyLoss()

    eval_config = {
        **classifier_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"]
        },
    }


    # --- Store original model parameters for weight tying ---
    original_params = {
        name: p.clone().detach().to(config["device"])
        for name, p in classifier.model_.named_parameters()
    }



    task_loss = 0
    tying_loss = 0

    # --- Feature embedding layer weights ---
    if FeatureEmbed:
        layer_weights = classifier.model_.encoder[5].layer.weight
        classifier.model_.encoder[5] = FeatureEmbedding(num_features=1, emsize=192, bias=False)
        if classifier.model_.encoder[5].layer is not None:
            # Replace linear layer with feature piece-wise embedding
            avg_weight = layer_weights.sum(dim=1)
            classifier.model_.encoder[5].layer.weight = torch.nn.Parameter(avg_weight.reshape(192,1))

    # --- Finetuning and Evaluation Loop ---
    print("--- 3. Starting Finetuning & Evaluation ---")
    for epoch in range(config["finetuning"]["epochs"] + 1):
        if epoch > 0:
            # Create a tqdm progress bar to iterate over the dataloader
            progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")
            for (
                X_train_batch,
                X_test_batch,
                y_train_batch,
                y_test_batch,
                cat_ixs,
                confs,
            ) in progress_bar:
                if len(np.unique(y_train_batch)) != len(np.unique(y_test_batch)):
                    continue  # Skip batch if splits don't have all classes

                optimizer.zero_grad()
                classifier.fit_from_preprocessed(
                    X_train_batch, y_train_batch, cat_ixs, confs
                )
                predictions = classifier.forward(X_test_batch, return_logits=True)



                # 1. Calculate the primary task loss (NLL for regression)
                task_loss = loss_function(predictions, y_test_batch.to(config["device"]))
                total_loss = task_loss  


                total_loss.backward()
                optimizer.step()

                # Set the postfix of the progress bar to show the current loss
                progress_bar.set_postfix(
                    task=f"{task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss:.4f}", 
                    # tying=f"{tying_loss.item() if isinstance(tying_loss, torch.Tensor) else tying_loss:.8f}"
                )


        epoch_roc, epoch_log_loss, epoch_accuracy = evaluate_model(
            classifier, eval_config, X_train, y_train, X_test, y_test
        )

        status = "Initial" if epoch == 0 else f"Epoch {epoch}"
        print(
            f"📊 {status} Evaluation | Test ROC: {epoch_roc:.9f}, Test Loss: {epoch_log_loss:.9f}, Test Accuracy: {epoch_accuracy:.9f}\n"
        )

    print("--- ✅ Finetuning Finished ---")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=1037)
    parser.add_argument("--FeatureEmbed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    main(id=args.dataset, 
         FeatureEmbed=args.FeatureEmbed, 
         LR=args.lr, 
         EPOCHS=args.epochs, 
         SEED=args.seed)


    
