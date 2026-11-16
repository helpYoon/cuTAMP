import numpy as np
import torch
from scipy.interpolate import CubicSpline
from cutamp.manifold_trajectory import _cubic_extrema

def test_exact_cubic_extrema_detect_between_knot_bound_violation():
    q=torch.tensor([[0.,0.],[1.,0.],[1.,0.],[0.,0.]],dtype=torch.float64)
    lo,hi=_cubic_extrema(q,.1)
    # A clamped cubic overshoots a [0,1] joint bound between equal top knots.
    assert hi[0]>1.1
    assert hi[1]==0 and lo[1]==0
    sample=CubicSpline(np.arange(4)*.1,q.numpy(),axis=0,bc_type='clamped')(np.linspace(0,.3,10001))
    assert np.all(lo.numpy()<=sample.min(0)+1e-12)
    assert np.all(hi.numpy()>=sample.max(0)-1e-12)
