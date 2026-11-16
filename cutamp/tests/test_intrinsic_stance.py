"""Regression: absolute sole costs must remain visible to native cspace optimization."""
import torch
from cutamp.com_polygon_cost import stance_pose_penalty

def test_stance_penalty_descent_and_rotation():
    target=torch.eye(4).repeat(2,1,1)
    poses=target.repeat(3,1,1,1).clone()
    poses[:,0,0,3]=.02
    poses.requires_grad_()
    cost=stance_pose_penalty(poses,target)
    gradient=torch.autograd.grad(cost.sum(),poses)[0]
    assert (stance_pose_penalty(poses-.1*gradient,target)<cost).all()
    # A rotated foot cannot evade the world-frame residual at zero translation.
    rotated=target.unsqueeze(0).clone()
    rotated[0,0,:2,:2]=torch.tensor([[0.,-1.],[1.,0.]])
    assert stance_pose_penalty(rotated,target).item()>.01
    assert stance_pose_penalty(target.unsqueeze(0),target).item()==0

def test_stance_penalty_cuda_replay_tracks_changed_input():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('CUDA required')
    target=torch.eye(4,device='cuda').repeat(2,1,1)
    poses=target.repeat(4,1,1,1).clone()
    output=torch.empty(4,device='cuda')
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3): output.copy_(stance_pose_penalty(poses,target))
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output.copy_(stance_pose_penalty(poses,target))
    poses[:,1,1,3]=.02
    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(output,torch.full_like(output,.0004))
