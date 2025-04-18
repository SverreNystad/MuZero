from dataclasses import dataclass
from torch import Tensor, tensor
from torch.nn import functional as F
import torch

@dataclass
class Frame:
    state: Tensor
    action: int

class FrameRingBuffer:
    """
    A simple ring buffer to store training data.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer: list[Frame] = []
        self.index = 0

    def add(self, item: Frame) -> None:
        """
        Add an item to the buffer. If the buffer is full, overwrite the oldest item.
        """
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.index] = item
        self.index = (self.index + 1) % self.size

    def fill(self, item: Frame) -> None:
        """
        Fill the buffer with the same item until it's full.
        """
        while len(self.buffer) < self.size:
            self.buffer.append(item)

    def get_all(self) -> list[Frame]:
        return self.buffer

    def full(self) -> bool:
        return len(self.buffer) >= self.size

def make_history_tensor(
    buffer: FrameRingBuffer,
) -> Tensor:
    """
    Turn the contents of a full FrameRingBuffer into the history tensor
    expected by MuZero‑like representation networks.

    Returns
    -------
    Tensor
        Shape: [(C*K)+K, H, W]   (channels_first)

        ├─ First  C*K  planes  : raw image frames, newest last
        └─ Last   K    planes  : one‑hot action planes
    """
    assert buffer.full(), "Ring buffer is not yet full"

    # 1. collect frames (oldest → newest) so we can concatenate
    frames = [f.state for f in buffer.get_all()]         # list[Tensor]
    img_tensor = torch.cat(frames, dim=1)  # concat on channel axis
                                           # shape -> [1, C*K, H, W]

    # 2. build one‑hot planes for every stored action
    actions = torch.tensor([f.action for f in buffer.get_all()],
                           dtype=torch.long, device=img_tensor.device)
                  # [K]
    # broadcast each scalar action id to an (H,W) image‑sized plane
    plane_list = [
        torch.full((1, *img_tensor.shape[2:]), a.item(),
                   dtype=img_tensor.dtype, device=img_tensor.device)
        for a in actions
    ]
    act_tensor = torch.cat(plane_list, dim=0)          # [K, H, W]
    act_tensor = act_tensor.unsqueeze(0)               # [1, K, H, W]

    # 3. concatenate images and action planes along channel dimension
    history = torch.cat([img_tensor, act_tensor], dim=1)
    # final shape  [1,(C*K)+K, H, W]

    return history

# if __name__ == "__main__":
#     # Example usage
#     max_actions = 32
#     buffer = FrameRingBuffer(max_actions)
#     buffer.fill(Frame(torch.zeros((1, 3, 512, 288)), 0))
#     buffer.add(Frame(torch.zeros((1, 3, 512, 288)), 1))
#     val = make_history_tensor(buffer, max_actions)
#     print(val.shape) # Should be (1, 128, 512, 288)
