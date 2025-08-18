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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor
from tabpfn_extensions.hpo import (
    TunedTabPFNRegressor,
)
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
import pandas as pd
import argparse

def get_effective_lr(base_lr, effective_batch_size):
    # TODO: maybe revise this logic
    return base_lr * np.sqrt(effective_batch_size)

def get_cosine_schedule_with_warmup(
    total_steps: float | int, warmup_steps: float | int, max_lr: float
):
    def lr_lambda(curr_step: int | float):
        if curr_step < warmup_steps:
            return max_lr * (curr_step / warmup_steps)
        else:
            decay_share = (curr_step - warmup_steps) / (total_steps - warmup_steps)
            return (
                max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * decay_share)))
                * max_lr
            )

    return lr_lambda

def set_lr(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr





LR_Schedulers = {
    "get_cosine_schedule_with_warmup": get_cosine_schedule_with_warmup,
}



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


def setup_regressor(config: dict) -> tuple[TabPFNRegressor, dict]:
    """Initializes the TabPFN regressor and its configuration."""
    print("--- 2. Model Setup ---")
    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 1,
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
        "inference_config": {
            "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None, None)
        },
    }
    regressor = TabPFNRegressor(
        **regressor_config, fit_mode="batched", differentiable_input=False
    )

    print(f"Using device: {config['device']}")
    print("----------------------\n")
    return regressor, regressor_config


def evaluate_regressor(
    regressor: TabPFNRegressor,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    """Evaluates the regressor's performance on the test set."""
    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    eval_regressor.fit(X_train, y_train)

    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()


    try:
        predictions = eval_regressor.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        mse, mae, r2, rmse = np.nan, np.nan, np.nan, np.nan

    return mse, mae, r2, rmse


def main(id: int = 0, 
         LAMBDA: float = 1e7, 
        #  learning_rate: float = 1.5e-7,
        #  which_loss: str = "norm_bardist", 
        #  which_y_test: str = "y_test_std"
         ):
    
    LR_List = [
            1e-08,
            3e-08,
            5e-08,
            8e-08,
            1e-07,
            3e-07,
            5e-07,
            8e-07,
            1e-06,
            4.999998054699972e-06,
            8.340500244230498e-06,
            1e-05,
            
        ]
    RMSE_LIST = []
    MSE_LIST = []

    

    for RAW_SPACE in [False, True]:
        # --- Finetuning and Evaluation Loop ---
        for lr in LR_List:

            

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
                "random_seed": 42,
                # The proportion of the dataset to allocate to the valid set for final evaluation.
                "valid_set_ratio": 0.3,
                # During evaluation, this is the number of samples from the training set given to the
                # model as context before it makes predictions on the test set.
                "n_inference_context_samples": 10000,
            }
            config["finetuning"] = {
                # The total number of passes through the entire fine-tuning dataset.
                "epochs": 50,
                # A small learning rate is crucial for fine-tuning to avoid catastrophic forgetting.
                "learning_rate": lr,
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
            config['finetuning']['l2_sp_lambda'] = LAMBDA

            # --- Logging ---
            print(f"Running finetuning for dataset: {id} \n")

            # --- Setup Data, Model, and Dataloader ---
            X_train, X_test, y_train, y_test = prepare_data(config, id)

            regressor, regressor_config = setup_regressor(config)

            splitter = partial(train_test_split, test_size=config["valid_set_ratio"])
            # Note: `max_data_size` corresponds to the finetuning `batch_size` in the config
            training_datasets = regressor.get_preprocessed_datasets(
                X_train, y_train, splitter, max_data_size=config["finetuning"]["batch_size"]
            )
            finetuning_dataloader = DataLoader(
                training_datasets,
                batch_size=config["finetuning"]["meta_batch_size"],
                collate_fn=meta_dataset_collator,
            )

            # Optimizer must be created AFTER get_preprocessed_datasets, which initializes the model
            optimizer = Adam(
                regressor.model_.parameters(), lr=config["finetuning"]["learning_rate"]
            )
            print(
                f"--- Optimizer Initialized: Adam, LR: {config['finetuning']['learning_rate']} ---\n"
            )

            # Create evaluation config, linking it to the master config
            eval_config = {
                **regressor_config,
                "inference_config": {
                    "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"]
                },
            }

            # --- Store original model parameters for weight tying ---
            original_params = {
                name: p.clone().detach().to(config["device"])
                for name, p in regressor.model_.named_parameters()
            }

            RMSE_SUB_LIST = []
            VAL_RMSE_SUB_LIST = []
            LOSS_SUB_LIST = []
            TYING_LOSS_SUB_LIST = []
            TOTAL_LOSS_SUB_LIST = []

            task_loss = 0
            tying_loss = 0

            print(f"LAMBDA: {config['finetuning']['l2_sp_lambda']} \n")

            # --- Finetuning and Evaluation Loop ---
            print("--- 3. Starting Finetuning & Evaluation ---")
            for epoch in range(config["finetuning"]["epochs"] + 1):
                if epoch > 0:
                    # Create a tqdm progress bar to iterate over the dataloader
                    progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")
                    for data_batch in progress_bar:
                        optimizer.zero_grad()
                        (
                            X_trains_p,
                            X_tests_p,
                            y_trains_p,
                            y_test_std,
                            cat_ixs,
                            confs,
                            norm_bardist,
                            bardist,
                            batch_x_raw,
                            batch_y_test_raw,
                        ) = data_batch

                        regressor.normalized_bardist_ = norm_bardist[0]
                        regressor.bardist_ = bardist[0]
                        regressor.fit_from_preprocessed(X_trains_p, y_trains_p, cat_ixs, confs)
                        logits, _, _ = regressor.forward(X_tests_p)

                        if RAW_SPACE:
                            loss_fn = norm_bardist[0]
                            y_target = batch_y_test_raw
                        else:
                            loss_fn = bardist[0]
                            y_target = y_test_std

                        

                        # 1. Calculate the primary task loss (NLL for regression)
                        task_loss = loss_fn(logits, y_target.to(device=config["device"], dtype=logits.dtype)).mean()
                        
                        # 2. Calculate the L2 weight tying loss (L2-SP regularization)
                        tying_loss = torch.tensor(0.0, device=config["device"])
                        l2_sp_lambda = config['finetuning']['l2_sp_lambda']
                        if l2_sp_lambda > 0:
                            for name, p_ft in regressor.model_.named_parameters():
                                if p_ft.requires_grad:
                                    tying_loss += torch.mean((p_ft - original_params[name]) ** 2) # Noah: mean
                            tying_loss = l2_sp_lambda * tying_loss * 0.5
                        
                        # 3. Combine losses and perform backpropagation
                        total_loss = task_loss + tying_loss 
                        

                        # Set the postfix of the progress bar to show the current loss
                        progress_bar.set_postfix(
                            task=f"{task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss:.4f}", 
                            tying=f"{tying_loss.item() if isinstance(tying_loss, torch.Tensor) else tying_loss:.8f}"
                        )
                        total_loss.backward()
                        optimizer.step()

                        

                        # if epoch == 1:
                            # print(f"executor: {regressor.executor_} \n")



                # Evaluation Step (runs before finetuning and after each epoch)
                mse, mae, r2, rmse = evaluate_regressor(
                    regressor, eval_config, X_train, y_train, X_test, y_test
                )
                RMSE_SUB_LIST.append(rmse)

                # Validation Evaluation
                if epoch > 0:
                    val_mse, val_mae, val_r2, val_rmse = evaluate_regressor(
                        regressor, eval_config, X_train, y_train, X_val, y_val
                    )
                    VAL_RMSE_SUB_LIST.append(val_rmse)
                
                    LOSS_SUB_LIST.append(task_loss.item())
                    TYING_LOSS_SUB_LIST.append(tying_loss.item())
                    TOTAL_LOSS_SUB_LIST.append(total_loss.item())

                    

                # TEST_RMSE_LIST.append(rmse)
            
                

                status = "Initial" if epoch == 0 else f"Epoch {epoch}"
                print(
                    f"📊 {status} Evaluation | Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test R2: {r2:.4f}, Test RMSE: {rmse:.4f}\n"
                )

            RMSE_LIST.append({'RAW_SPACE': RAW_SPACE,
                                'lr': lr,
                                'loss': LOSS_SUB_LIST,
                                'tying_loss': TYING_LOSS_SUB_LIST,
                                'total_loss': TOTAL_LOSS_SUB_LIST,
                                'rmse': RMSE_SUB_LIST,
                                'val_rmse': VAL_RMSE_SUB_LIST,
                                })



    print("--- ✅ Finetuning Finished ---")
    
    # Save the figure to a file
    os.makedirs(f'finetune_0817_yandex', exist_ok=True)

    RMSE_DF = pd.DataFrame(RMSE_LIST)
    RMSE_DF.to_csv(f"finetune_0817_yandex/RMSE_LIST_yandex_{id}_{np.log10(LAMBDA)}.csv", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="california")
    parser.add_argument("--l2_sp_lambda", type=float, default=1e7)
    
    args = parser.parse_args()
    main(args.dataset, args.l2_sp_lambda)


    
