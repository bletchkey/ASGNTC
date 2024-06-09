import torch
import time
import subprocess
import logging
from typing import List

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
        self.__primary_device       = self.__get_primary_device()
        self.__secondary_device     = self.__get_secondary_device()


    @property
    def n_gpus(self) -> int:
        """Returns the total number of GPUs available."""
        return self.__n_gpus

    @property
    def default_device(self) -> torch.device:
        """Returns the default device"""
        return self.__default_device

    @property
    def primary_device(self) -> torch.device:
        """Returns the primary device"""
        return self.__primary_device

    @property
    def secondary_device(self) -> torch.device:
        """Returns the secondary device"""
        return self.__secondary_device

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

        if torch.cuda.is_available:

            for i in range(self.__n_gpus):
                torch.cuda.set_device(i)
                device           = torch.device(i)
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


    def set_default_device(self, device):
        """
        Function to set the default device for training.

        Args:
            device (torch.device): The device to set as the default device.

        """

        self.__default_device = torch.device(device)


    def __query_gpu_memory(self):
        """
        Uses nvidia-smi to get the free memory on each GPU in MB.

        """

        try:
            smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], encoding='utf-8')

            # Parse the output to get a list of free memory values for each GPU
            free_memory = [int(x) for x in smi_output.strip().split('\n')]

            return free_memory

        except Exception as e:
            logging.error(f"Failed to query GPU free memory: {str(e)}")
            return None


    def __get_default_device(self, use_benchmark=False):
        """
        Determines the optimal device

        Returns:
            torch.device: The device selected for training operations.
        """

        if not torch.cuda.is_available():
            return torch.device("cpu")

        if use_benchmark == True:
            gpus_free_memory    = self.__get_all_gpus_free_memory()
            gpus_benchmark_time = {}

            for key in gpus_free_memory.keys():
                benchmark_time = self.__benchmark_gpu(torch.device(f"cuda:{key}"))
                if benchmark_time != float('inf'):  # Only consider devices that successfully complete the benchmark
                    gpus_benchmark_time[key] = benchmark_time

            # Select device based on benchmark or fallback to a device with maximum free memory
            if gpus_benchmark_time:
                selected_device = min(gpus_benchmark_time, key=lambda k: (-gpus_free_memory[k], gpus_benchmark_time[k]))
            else:
                selected_device = max(gpus_free_memory, key=gpus_free_memory.get)
        else:
            selected_device = self.__get_device_with_max_free_memory()

        logging.debug(f"Selected device: {selected_device}")

        torch.cuda.set_device(selected_device)

        return torch.device(selected_device)


    def __get_primary_device(self) -> torch.device:
        """
        Determines the primary device

        Returns:
            torch.device: The primary device selected for training operations.
        """

        if not torch.cuda.is_available():
            return torch.device("cpu")

        return torch.device(self.__default_device.index)


    def __get_secondary_device(self) -> torch.device:
        """
        Determines the secondary device

        Returns:
            torch.device: The secondary device selected for training operations.
        """

        if not torch.cuda.is_available() or self.__n_gpus <= 1:
            return torch.device("cpu")

        gpus = self.__get_all_gpus_free_memory()
        gpus.pop(self.__default_device.index)

        return torch.device(max(gpus, key=gpus.get))


    def __benchmark_gpu(self, device):
        try:
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()

            start_time = time.time()

            tensor_size = (100, 1024, 1024)
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
        """
        Function that returns the free memory for a given device.

        """
        try:
            torch.cuda.set_device(device)
            id = torch.cuda.current_device()

            free_memory = self.__query_gpu_memory()

            if free_memory is None or not free_memory:
                logging.warning("Unable to get GPU memory details, falling back to CPU.")

            return free_memory[id]

        except Exception as e:
            logging.error(f"Error getting free memory for device {device}: {str(e)}")
            return None


    def __get_device_with_max_free_memory(self):
        """
        Function that returns the device with the maximum free memory.

        """
        if not torch.cuda.is_available():
            return torch.device("cpu")

        max_memory_free = 0
        selected_device = 0

        for i in range(self.__n_gpus):
            free_memory = self.__get_device_free_memory(i)

            if free_memory > max_memory_free:
                max_memory_free = free_memory
                selected_device = i

        return torch.device(selected_device)


    def __get_all_gpus_free_memory(self):
        """
        Function that returns the free memory for all GPUs.

        """
        free_memory = {}

        for i in range(self.__n_gpus):
            free_memory[i] = self.__get_device_free_memory(i)

        return free_memory


    def __get_balanced_gpu_indices(self, tolerance=0.25) -> List[int]:
        """
        Identifies GPUs that are sufficiently balanced in terms of memory and core count compared to the default device.

        Args:
            tolerance (float, optional): The maximum allowed deviation in memory and core counts from the default device to be considered balanced. Defaults to 0.25.

        Returns:
            List[int]: A list of indices for GPUs that meet the balance criteria.

        """
        if not torch.cuda.is_available() or self.__n_gpus <= 1:
            return []

        default_device_index   = torch.cuda.current_device()
        default_gpu_properties = torch.cuda.get_device_properties(default_device_index)

        balanced_gpus = []

        for i in range(self.__n_gpus):

            if i == default_device_index:
                continue

            gpu_properties = torch.cuda.get_device_properties(i)
            memory_ratio   = gpu_properties.total_memory / default_gpu_properties.total_memory
            cores_ratio    = gpu_properties.multi_processor_count / default_gpu_properties.multi_processor_count

            # Calculate absolute differences in ratios from 1
            memory_diff = abs(memory_ratio - 1)
            cores_diff  = abs(cores_ratio - 1)

            logging.debug(f"GPU {i} - Memory ratio: {memory_ratio:.2f}, Cores ratio: {cores_ratio:.2f} in comparison to the default device [GPU {self.__default_device}].")

            logging.debug(f"GPU {i} - Memory diff: {memory_diff:.2f}, Cores diff: {cores_diff:.2f}.")

            # Check if the differences are within the specified tolerance
            if memory_diff <= tolerance and cores_diff <= tolerance:
                balanced_gpus.append(i)
                logging.debug(f"GPU {i} is considered balanced with respect to the default device [GPU {self.__default_device}].")

        return balanced_gpus

