import torch
import logging

class DeviceManager:
    """
    A manager class for handling device selection and balancing across GPUs for PyTorch models.

    This class automatically identifies the best default device based on available GPUs and their memory status.
    It also provides functionalities to retrieve balanced GPU indices for distributed training tasks,
    ensuring optimal utilization of available hardware resources.

    Attributes:
        n_gpus (int): The number of GPUs available on the system.
        default_device (torch.device): The selected default device for training.
        balanced_gpu_indices (List[int]): Indices of GPUs considered balanced for distributed tasks.
        n_balanced_gpus (int): The number of GPUs that are considered balanced.
    """

    def __init__(self) -> None:
        """
        Initializes the DeviceManager by detecting available GPUs and selecting the optimal training device.

        """

        self.__n_gpus               = torch.cuda.device_count()
        self.__default_device       = self.__get_default_device()
        self.__balanced_gpu_indices = self.__get_balanced_gpu_indices()
        self.__n_balanced_gpus      = len(self.__balanced_gpu_indices)


    @property
    def n_gpus(self) -> int:
        """Returns the total number of GPUs available."""
        return self.__n_gpus

    @property
    def default_device(self) -> torch.device:
        """Returns the default device selected for training."""
        return self.__default_device

    @property
    def balanced_gpu_indices(self) -> list:
        """Returns a list of GPU indices considered balanced based on memory and cores."""
        return self.__balanced_gpu_indices

    @property
    def n_balanced_gpus(self) -> int:
        """Returns the number of GPUs considered balanced."""
        return self.__n_balanced_gpus


    def __get_default_device(self):
        """
        Determines the optimal device, preferring the GPU with the most free memory if available.

        Returns:
            torch.device: The device selected for training operations.
        """

        if not torch.cuda.is_available():
            return torch.device("cpu")

        max_memory_free = 0
        selected_device = 0

        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            memory_stats = torch.cuda.memory_stats()
            memory_free = memory_stats["allocated_bytes.all.peak"] - memory_stats["allocated_bytes.all.current"]
            if memory_free > max_memory_free:
                max_memory_free = memory_free
                selected_device = i

        return torch.device(f"cuda:{selected_device}")


    def __get_balanced_gpu_indices(self, threshold=0.75):
        """
        Identifies GPUs that are sufficiently balanced in terms of memory and core count compared to the default device.

        Args:
            threshold (float, optional): The minimum ratio of memory and cores compared to the default device to be considered balanced. Defaults to 0.75.

        Returns:
            List[int]: A list of indices for GPUs that meet the balance criteria.

        """

        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            return []

        id_default_device      = torch.cuda.current_device()
        default_gpu_properties = torch.cuda.get_device_properties(id_default_device)
        balanced_gpus          = []

        for i in range(torch.cuda.device_count()):
            if i == id_default_device:
                continue
            gpu_properties = torch.cuda.get_device_properties(i)
            memory_ratio   = gpu_properties.total_memory / default_gpu_properties.total_memory
            cores_ratio    = gpu_properties.multi_processor_count / default_gpu_properties.multi_processor_count

            if memory_ratio >= threshold and cores_ratio >= threshold:
                balanced_gpus.append(i)
                logging.debug(f"GPU {i} is balanced with respect to the default device [GPU {id_default_device}].")

        return balanced_gpus


    def clear_resources(self, threshold=0.75):
        """
        Function for cleaning up the resources.
        If the device used for training is a GPU, the CUDA cache is cleared.

        """

        if torch.cuda.is_available():
            device           = self.__default_device
            allocated_memory = torch.cuda.memory_allocated(device)
            reserved_memory  = torch.cuda.memory_reserved(device)
            logging.debug(f"Allocated memory on {device}: {allocated_memory / (1024**3):.2f} GB")
            logging.debug(f"Reserved memory on {device}: {reserved_memory / (1024**3):.2f} GB")

            if allocated_memory / reserved_memory > threshold:
                logging.debug(f"Clearing CUDA cache on {device}")
                torch.cuda.empty_cache()

                allocated_memory = torch.cuda.memory_allocated(device)
                reserved_memory  = torch.cuda.memory_reserved(device)

                logging.debug(f"Allocated memory on {device}: {allocated_memory / (1024**3):.2f} GB")
                logging.debug(f"Reserved memory on {device}: {reserved_memory / (1024**3):.2f} GB")

