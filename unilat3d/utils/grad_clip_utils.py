from typing import *
import torch
import numpy as np
import torch.utils


class AdaptiveGradClipper:
    
    def __init__(
        self,
        max_norm=None,
        clip_percentile=95.0,
        buffer_size=1000,
    ):
        self.max_norm = max_norm
        self.clip_percentile = clip_percentile
        self.buffer_size = buffer_size
        
        self._grad_norm = np.zeros(buffer_size, dtype=np.float32)
        self._max_norm = max_norm
        self._buffer_ptr = 0
        self._buffer_length = 0

    def __repr__(self):
        return f'AdaptiveGradClipper(max_norm={self.max_norm}, clip_percentile={self.clip_percentile})'
        
    def state_dict(self):
        return {
            'grad_norm': self._grad_norm,
            'max_norm': self._max_norm,
            'buffer_ptr': self._buffer_ptr,
            'buffer_length': self._buffer_length,
        }

    def load_state_dict(self, state_dict):
        self._grad_norm = state_dict['grad_norm']
        self._max_norm = state_dict['max_norm']
        self._buffer_ptr = state_dict['buffer_ptr']
        self._buffer_length = state_dict['buffer_length']

    def log(self):
        return {
            'max_norm': self._max_norm,
        }

    def __call__(self, parameters, norm_type=2.0, error_if_nonfinite=False, foreach=None):
        
        max_norm = self._max_norm if self._max_norm is not None else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=foreach)
        
        if torch.isfinite(grad_norm):
            self._grad_norm[self._buffer_ptr] = grad_norm
            self._buffer_ptr = (self._buffer_ptr + 1) % self.buffer_size
            self._buffer_length = min(self._buffer_length + 1, self.buffer_size)
            if self._buffer_length == self.buffer_size:
                self._max_norm = np.percentile(self._grad_norm, self.clip_percentile)
                self._max_norm = min(self._max_norm, self.max_norm) if self.max_norm is not None else self._max_norm
        
        return grad_norm