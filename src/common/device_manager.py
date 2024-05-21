import torch
import time
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


    def __get_default_device(self):
        """
        Determines the optimal device

        Returns:
            torch.device: The device selected for training operations.
        """

        if not torch.cuda.is_available():
            return torch.device("cpu")

        # Get gpus free memory status
        gpus_free_memory = self.__get_all_gpus_free_memory()

        # Check benchmarking
        gpus_benchmark_time = {}

        for key in gpus_free_memory.keys():
            gpus_benchmark_time[key] = self.__benchmark_gpu(torch.device(f"cuda:{key}"))

        # Select the GPU with the most free memory or best performance
        selected_device = min(gpus_benchmark_time, key=lambda k: (-gpus_free_memory[k], gpus_benchmark_time[k]))

        logging.debug(f"Selected device: {selected_device}")

        return torch.device(f"cuda:{selected_device}")


    def __benchmark_gpu(self, device):
        try:
            torch.cuda.set_device(device)
            start_time = time.time()

            tensor_size = (1024, 1024, 1024)  # 1 GB tensor
            x = torch.randn(tensor_size, device=device)
            y = torch.randn(tensor_size, device=device)
            z = x * y

            torch.cuda.synchronize()
            end_time = time.time()

            return end_time - start_time

        except Exception as e:
            logging.error(f"Error benchmarking device {device}: {str(e)}")
            return float('inf')


    def __get_device_free_memory(self, device):
        try:
            torch.cuda.set_device(device)

            total_memory     = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            reserved_memory  = torch.cuda.memory_reserved(device)
            free_memory      = total_memory - max(allocated_memory, reserved_memory)

            logging.debug(f"GPU {device} - Total memory: {total_memory / (1024**3):.2f} GB")
            logging.debug(f"GPU {device} - Allocated memory: {allocated_memory / (1024**3):.2f} GB")
            logging.debug(f"GPU {device} - Reserved memory: {reserved_memory / (1024**3):.2f} GB")
            logging.debug(f"GPU {device} - Free memory: {free_memory / (1024**3):.2f} GB")

            return free_memory

        except Exception as e:
            logging.error(f"Error getting free memory for device {device}: {str(e)}")
            return None


    def __get_device_with_max_free_memory(self):
        if not torch.cuda.is_available():
            return torch.device("cpu")

        max_memory_free = 0
        selected_device = 0

        for i in range(self.__n_gpus):
            free_memory = self.__get_device_free_memory(i)

            if free_memory > max_memory_free:
                max_memory_free = free_memory
                selected_device = i

        return torch.device(f"cuda:{selected_device}")


    def __get_all_gpus_free_memory(self):
        free_memory = {}

        for i in range(self.__n_gpus):
            free_memory[i] = self.__get_device_free_memory(i)

        return free_memory


    def __get_balanced_gpu_indices(self, threshold=0.75):
        """
        Identifies GPUs that are sufficiently balanced in terms of memory and core count compared to the default device.

        Args:
            threshold (float, optional): The minimum ratio of memory and cores compared to the default device to be considered balanced. Defaults to 0.75.

        Returns:
            List[int]: A list of indices for GPUs that meet the balance criteria.

        """

        if not torch.cuda.is_available() or self.__n_gpus <= 1:
            return []

        torch.cuda.set_device(self.__default_device)

        id_default_device      = torch.cuda.current_device()
        default_gpu_properties = torch.cuda.get_device_properties(id_default_device)
        balanced_gpus          = []

        for i in range(self.__n_gpus):
            if i == id_default_device:
                continue
            gpu_properties = torch.cuda.get_device_properties(i)
            memory_ratio   = gpu_properties.total_memory / default_gpu_properties.total_memory
            cores_ratio    = gpu_properties.multi_processor_count / default_gpu_properties.multi_processor_count

            if memory_ratio >= threshold and cores_ratio >= threshold:
                balanced_gpus.append(i)
                logging.debug(f"GPU {i} is balanced with respect to the default device [GPU {id_default_device}].")

        return balanced_gpus

