import torch
import torch.nn as nn

from test import test

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
if __name__ == "__main__":
    test()