import torch
import torch.nn as nn
from .optimal_transport import OptimalTransportConditionalFlowMatching

class RectifiedFlow(OptimalTransportConditionalFlowMatching):
    """
    Rectified Flow (Reflow).
    
    References:
    Liu, X., et al. (2022). "Flow Straight and Fast: Learning to Generate with Rectified Flow". ICLR 2023.
    
    Reflow 1-step is essentially OT-CFM.
    Reflow 2-step uses (Z, T(Z)) pairs where T(Z) is the generated sample from the 1-step model.
    This module supports the general framework relative to OT-CFM.
    """
    def __init__(self, sigma=0.0):
        super().__init__(sigma=sigma)
        
    # Reflow shares the same loss objective as OT-CFM (straight paths),
    # but the PAIRING (x0, x1) changes during iterative refinement.
    # The logic for "Reflow" is typically handled in the training loop 
    # (generate pairs -> retrain), or by simply enforcing OT which is the theoretical limit of Reflow.
    # We will stick to OT-CFM as "1-Rectified Flow" in this codebase for clarity,
    # as 1-Rectified is equivalent to OT-CFM.
    
    pass
