import numpy as np
import torch
import torch.nn as nn


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class WavelenghtSinusoidalEmbedding(nn.Module):
    def __init__(self, min_wavelength, max_wavelength, embedding_dim):
        """
        :param min_wavelength: The minimum wavelength for the embedding.
        :param max_wavelength: The maximum wavelength for the embedding.
        :param embedding_dim: The dimension of the output embedding for each wavelength. 
        The final output embedding will have a dimension of embedding_dim * 2 (sin and cos).
        """
        super(WavelenghtSinusoidalEmbedding, self).__init__()
        self.min_wl = min_wavelength
        self.max_wl = max_wavelength
        self.embedding_dim = embedding_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        
        # The inv_freq is registered as a buffer 
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        

    def forward(self, wavelengths):
        """
        :param wavelengths: A 1d tensor of size (x,)
        :return: Positional Encoding Matrix of size (x, embedding_dim)
        """
        if len(wavelengths.shape) != 1:
            raise RuntimeError("The input tensor has to be 1d!")

        if self.cached_penc is not None and self.cached_penc.shape == wavelengths.shape:
            return self.cached_penc

        self.cached_penc = None
        x = wavelengths.size(0)
        # normalize the wavelengths to be between 0 and 1
        wavelengths = wavelengths.float()
        wavelengths = (wavelengths - self.min_wl) / (self.max_wl - self.min_wl)

        # Create the positional encoding matrix
        sin_inp_x = torch.einsum("i,j->ij", wavelengths, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros(
            (x, self.embedding_dim),
            device=wavelengths.device,
        )
        emb[:, : self.embedding_dim] = emb_x

        self.cached_penc = emb
        return self.cached_penc

class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size(), penc.size()
        )
        return tensor + penc.to(tensor.device)
