import torch


def toroidal_Conv2d(x, conv2d=None, padding=1):
        """
        Function for adding custom padding (toroidal) to a convolutional layer
        """
        x = __add_toroidal_padding(x, padding)
        x = conv2d(x)

        return x


def __add_toroidal_padding(x: torch.Tensor, padding: int=1) -> torch.Tensor:
    """
    Function for adding toroidal padding

    Args:
        x (torch.Tensor): The tensor to add padding to

    Returns:
        x (torch.Tensor): The tensor with toroidal padding

    """
    if x.dim() != 4:
        raise RuntimeError(f"Expected 4D tensor, got {x.dim()}")

    if padding <= 0:
        return x

    x = torch.cat([x[:, :, -padding:], x, x[:, :, :padding]], dim=2)
    x = torch.cat([x[:, :, :, -padding:], x, x[:, :, :, :padding]], dim=3)

    return x

