from pathlib import Path
from typing import Dict
import logging
import torch
import os
import re
import numpy as np
import traceback

from configs.constants import *

from src.common.device_manager        import DeviceManager
from src.gol_pred_sys.dataset_manager import DatasetManager
from src.common.predictor             import Predictor_Baseline

from src.gol_pred_sys.utils.helpers import get_config_from_batch
from src.common.utils.scores        import prediction_score


def __extract_checkpoint_index(filename):
    try:
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            raise ValueError("No checkpoint index found in filename")
    except Exception as e:
        logging.error(f"Error extracting checkpoint index from {filename}: {e}")
        return None


def get_prediction_score(model_folder_path: Path, checkpoint_index:int = None):

    device_manager  = DeviceManager()
    device          = device_manager.default_device
    dataset_manager = DatasetManager()
    dataloader_test = dataset_manager.get_dataloader(TEST, P_BATCH_SIZE, shuffle=False)

    if checkpoint_index is None:
        model_checkpoints = sorted(os.listdir(model_folder_path / "checkpoints"), key=__extract_checkpoint_index)
    else:
        model_checkpoints = [f"predictor_{checkpoint_index}.pth.tar"]

    scores = []

    for checkpoint in model_checkpoints:
        try:
            checkpoint_path = model_folder_path / "checkpoints" / checkpoint

            checkpoint = torch.load(checkpoint_path)

            model = checkpoint[CHECKPOINT_MODEL_ARCHITECTURE_KEY]

            model.load_state_dict(checkpoint[CHECKPOINT_MODEL_STATE_DICT_KEY])
            model.to(device)
            model.eval()

            total_score = 0

            with torch.no_grad():
                for batch_count, (batch, _) in enumerate(dataloader_test, start=1):
                    input  = get_config_from_batch(batch,
                                                   checkpoint[CHECKPOINT_P_INPUT_TYPE],
                                                   device)
                    target = get_config_from_batch(batch,
                                                   checkpoint[CHECKPOINT_P_TARGET_TYPE],
                                                   device)

                    predicted = model(input)
                    score     = prediction_score(predicted, target)

                    total_score       += score
                    running_avg_score  = total_score / batch_count

            scores.append(running_avg_score)

            logging.debug(f"Checkpoint {checkpoint_path} - Prediction score: {running_avg_score*100:.2f}%")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing checkpoint {checkpoint_path}: {e}")
            traceback.print_exc()

    if len(scores) == 1:
        return scores[0]

    return scores


def get_prediction_score_n_cells_initial(checkpoint_path: Path) -> Dict[int, float]:
    try:
        device_manager = DeviceManager()
        device = device_manager.default_device
        dataset_manager = DatasetManager()
        dataloader_test = dataset_manager.get_dataloader(TEST, P_BATCH_SIZE, shuffle=False)

        checkpoint = torch.load(checkpoint_path)

        model = checkpoint[CHECKPOINT_MODEL_ARCHITECTURE_KEY]
        model.load_state_dict(checkpoint[CHECKPOINT_MODEL_STATE_DICT_KEY])
        model.to(device)
        model.eval()

        avg_score_each_n_cells = {}

        with torch.no_grad():
            for batch, metadata in dataloader_test:
                input  = get_config_from_batch(batch, checkpoint[CHECKPOINT_P_INPUT_TYPE], device)
                target = get_config_from_batch(batch, checkpoint[CHECKPOINT_P_TARGET_TYPE], device)

                predicted = model(input)
                score     = prediction_score(predicted, target)
                n_cells   = metadata[META_N_CELLS_INITIAL]

                if n_cells not in avg_score_each_n_cells:
                    avg_score_each_n_cells[n_cells] = []

                avg_score_each_n_cells[n_cells].append(score)

        avg_score_each_n_cells = {k: np.mean(v) for k, v in avg_score_each_n_cells.items()}

        return avg_score_each_n_cells

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

