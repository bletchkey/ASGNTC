import logging
import torch

from configs.constants import *
from configs.paths     import CONFIG_DIR, TRAININGS_DIR, \
                              TRAINED_MODELS_DIR, TRAININGS_PREDICTOR_DIR
from configs.paths     import DATASET_DIR

from src.common.device_manager        import DeviceManager
from src.gol_pred_sys.dataset_manager import DatasetManager

from src.gol_pred_sys.utils.helpers import get_config_from_batch
from src.common.utils.scores        import prediction_accuracy_bins, \
                                           prediction_accuracy_tolerance,\
                                           prediction_accuracy_ssim


def get_accuracy(checkpoint_path):

    device_manager = DeviceManager()
    device         = device_manager.default_device

    dataset_manager = DatasetManager()
    dataloader_test = dataset_manager.get_dataloader(TEST, P_BATCH_SIZE, shuffle=False)

    checkpoint = torch.load(checkpoint_path)

    model = checkpoint["model"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for batch_count, (batch, _) in enumerate(dataloader_test, start=1):
            input  = get_config_from_batch(batch,
                                           checkpoint["config_type_pred_input"],
                                           device)
            target = get_config_from_batch(batch,
                                           checkpoint["config_type_pred_target"],
                                           device)

            predicted = model(input)
            # accuracy  = prediction_accuracy_bins(predicted, target)
            # accuracy = prediction_accuracy_tolerance(predicted, target, 0.2)
            accuracy = prediction_accuracy_ssim(predicted, target)

            total_accuracy += accuracy
            running_avg_accuracy = total_accuracy / batch_count

            logging.debug(f"Batch {batch_count} - Accuracy: {100*accuracy:.1f}% - Running average: {running_avg_accuracy*100:.1f}%")

    del model
    torch.cuda.empty_cache()

    return running_avg_accuracy

