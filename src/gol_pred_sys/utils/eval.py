import logging
import torch
import os
import re
import traceback

from configs.constants import *

from src.common.device_manager        import DeviceManager
from src.gol_pred_sys.dataset_manager import DatasetManager
from src.common.predictor             import Predictor_Baseline

from src.gol_pred_sys.utils.helpers import get_config_from_batch
from src.common.utils.scores        import prediction_accuracy_bins, \
                                           prediction_accuracy_tolerance,\
                                           prediction_accuracy_ssim,\
                                           prediction_accuracy


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


def  get_accuracies(model_folder_path):

    device_manager  = DeviceManager()
    device          = device_manager.default_device
    dataset_manager = DatasetManager()
    dataloader_test = dataset_manager.get_dataloader(TEST, P_BATCH_SIZE, shuffle=False)

    model_checkpoints = sorted(os.listdir(model_folder_path / "checkpoints"), key=__extract_checkpoint_index)

    accuracies = []

    for checkpoint in model_checkpoints:
        try:
            checkpoint_path = model_folder_path / "checkpoints" / checkpoint

            checkpoint = torch.load(checkpoint_path)

            model = checkpoint[CHECKPOINT_MODEL_ARCHITECTURE_KEY]

            model.load_state_dict(checkpoint[CHECKPOINT_MODEL_STATE_DICT_KEY])
            model.to(device)
            model.eval()

            total_accuracy = 0

            with torch.no_grad():
                for batch_count, (batch, _) in enumerate(dataloader_test, start=1):
                    input  = get_config_from_batch(batch,
                                                   checkpoint[CHECKPOINT_P_INPUT_TYPE],
                                                   device)
                    target = get_config_from_batch(batch,
                                                   checkpoint[CHECKPOINT_P_TARGET_TYPE],
                                                   device)

                    predicted = model(input)
                    accuracy  = prediction_accuracy(predicted, target, 0.2)

                    total_accuracy += accuracy
                    running_avg_accuracy = total_accuracy / batch_count

                    logging.debug(f"Batch {batch_count} - Accuracy: {100*accuracy:.1f}% - Running average: {running_avg_accuracy*100:.1f}%")


            accuracies.append(running_avg_accuracy)

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing checkpoint {checkpoint_path}: {e}")
            traceback.print_exc()

    return accuracies

