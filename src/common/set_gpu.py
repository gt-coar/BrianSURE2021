import torch

def set_gpu():
    torch.backends.cuda.matmul.allow_tf32 = True # setting default gpu for matmul on float 32 tensor
    torch.backends.cudnn.allow_tf32 = True # setting default gpu for float 32 tensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor) # set cuda float tensor as default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device