import torch


class DeviceManager:

    def __init__(self) -> None:
        self.__n_gpus = torch.cuda.device_count()
        self.__default_device = self.__get_default_device()
        self.__balanced_gpu_indices = self.__get_balanced_gpu_indices()
        self.__n_balanced_gpus = len(self.__balanced_gpu_indices)

    @property
    def n_gpus(self):
        return self.__n_gpus

    @property
    def default_device(self):
        return self.__default_device

    @property
    def balanced_gpu_indices(self):
        return self.__balanced_gpu_indices

    @property
    def n_balanced_gpus(self):
        return self.__n_balanced_gpus


    """
    Function for choosing the device to use for training.
    If a GPU is available, the GPU with the most free memory is selected.

    Returns:
        device (torch.device): The device to use for training.
    """
    def __get_default_device(self):
        if not torch.cuda.is_available():
            return torch.device("cpu")

        device_specs = []

        for i in range(torch.cuda.device_count()):
            gpu_properties = torch.cuda.get_device_properties(i)
            memory_free = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
            device_specs.append((i, gpu_properties.total_memory, memory_free))

        selected_device = max(device_specs, key=lambda x: x[2])[0]

        return torch.device(f"cuda:{selected_device}")


    """
    Get GPU indices that are balanced compared to the default device that has been chosen.

    Args:
        threshold (float): The minimum ratio of memory and cores compared to the default device to be considered balanced.

    Returns:
        list: A list of GPU indices that are balanced.
    """
    def __get_balanced_gpu_indices(self, threshold=0.75):
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            return []

        id_default_device = torch.cuda.current_device()
        default_gpu_properties = torch.cuda.get_device_properties(id_default_device)
        balanced_gpus = []

        for i in range(torch.cuda.device_count()):
            gpu_properties = torch.cuda.get_device_properties(i)
            memory_ratio = gpu_properties.total_memory / default_gpu_properties.total_memory
            cores_ratio = gpu_properties.multi_processor_count / default_gpu_properties.multi_processor_count

            if memory_ratio >= threshold and cores_ratio >= threshold and i != id_default_device:
                balanced_gpus.append(i)

        return balanced_gpus

