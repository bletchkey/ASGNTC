from pathlib import Path
import logging
import torch
import os
import re
import numpy as np
import traceback

from configs.constants import *

from src.common.device_manager        import DeviceManager

from src.gol_adv_sys.utils.helpers         import generate_initial_config, \
                                                  get_initialized_initial_config

from src.common.utils.simulation_functions import simulate_config


def __extract_checkpoint_index(filename):
    try:
        match = re.search(r"(\d+)", filename)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("No checkpoint index found in filename")
    except Exception as e:
        logging.error(f"Error extracting checkpoint index from {filename}: {e}")
        return None


def get_generator_eval_stats(folder_path: Path, checkpoint_index:int = None):

    device_manager  = DeviceManager()
    device          = device_manager.default_device

    if checkpoint_index is None:
        checkpoint_files = [f for f in os.listdir(folder_path / "checkpoints") if 'generator_' in f]
        model_checkpoints = sorted(checkpoint_files, key=__extract_checkpoint_index)
    else:
        model_checkpoints = [f"generator_{checkpoint_index}.pth.tar"]

    stats ={"n_cells_initial" : [],
            "n_cells_final"   : [],
            "period"          : [],
            "transient_phase" : [],
            "generator_name"  : None,
            "pred_target"     : None}

    for checkpoint in model_checkpoints:
        try:
            checkpoint_path = folder_path / "checkpoints" / checkpoint

            checkpoint = torch.load(checkpoint_path)

            stats["pred_target"]    = checkpoint[CHECKPOINT_P_TARGET_TYPE]
            stats["generator_name"] = checkpoint[CHECKPOINT_MODEL_NAME_KEY]

            model = checkpoint[CHECKPOINT_MODEL_ARCHITECTURE_KEY]

            model.load_state_dict(checkpoint[CHECKPOINT_MODEL_STATE_DICT_KEY])
            model.to(device)
            model.eval()

            temp_stats = {"n_cells_initial" : [],
                          "n_cells_final"   : [],
                          "period"          : [],
                          "transient_phase" : []}
            with torch.no_grad():
                for i in range(NUM_BATCHES_GEN_EVAL):

                    generated_config = generate_initial_config(model, device)
                    initial_config   = get_initialized_initial_config(generated_config, INIT_CONFIG_INITIAL_SIGN)
                    sim_results      = simulate_config(config=initial_config, topology=TOPOLOGY_TOROIDAL,
                                                       steps=N_SIM_STEPS, device=device)

                    temp_stats["n_cells_initial"].append(sim_results["n_cells_initial"].cpu())
                    temp_stats["n_cells_final"].append(sim_results["n_cells_final"].cpu())
                    temp_stats["period"].append(sim_results["period"].cpu())
                    temp_stats["transient_phase"].append(sim_results["transient_phase"].cpu())

                    print(f"Batch {i+1}/{NUM_BATCHES_GEN_EVAL} stats retrieved successfully.")

                # Convert lists to numpy arrays
                temp_stats["n_cells_initial"] = np.array([x.numpy() for x in temp_stats["n_cells_initial"]])
                temp_stats["n_cells_final"] = np.array([x.numpy() for x in temp_stats["n_cells_final"]])
                temp_stats["period"] = np.array([x.numpy() for x in temp_stats["period"]])
                temp_stats["transient_phase"] = np.array([x.numpy() for x in temp_stats["transient_phase"]])

                stats["n_cells_initial"].append(np.mean(temp_stats["n_cells_initial"]))
                stats["n_cells_final"].append(np.mean(temp_stats["n_cells_final"]))
                stats["period"].append(np.mean(temp_stats["period"]))
                stats["transient_phase"].append(np.mean(temp_stats["transient_phase"]))

                print(f"Checkpoint {checkpoint_path} average stats retrieved successfully.")

                print(stats)

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing checkpoint {checkpoint_path}: {e}")
            traceback.print_exc()

    return stats

