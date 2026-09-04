from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
                                                                                      
                                                                                             
                                                                                     
        kwargs.pop("neg_cond", None)
        kwargs.pop("cfg_strength", None)
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred
