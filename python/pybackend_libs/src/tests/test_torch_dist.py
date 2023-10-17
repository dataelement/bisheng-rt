import torch
from accelerate import Accelerator


def main():
    print('Cuda support:', torch.cuda.is_available(), ':',
          torch.cuda.device_count(), 'devices')
    accelerator = Accelerator()
    print(accelerator.state)


if __name__ == '__main__':
    main()
