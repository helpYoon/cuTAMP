from types import SimpleNamespace
import torch,pytest
from cutamp.anatomical_comfort import residual

def test_anatomical_forward_backward_cuda_graph_replay():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    world=SimpleNamespace(model=SimpleNamespace(joint_names=['x','y','z','pitch','roll','yaw','waist_pitch_joint','waist_roll_joint','waist_yaw_joint']),stance_soles=torch.eye(4,device='cuda').repeat(2,1,1))
    q=torch.zeros(64,9,device='cuda',requires_grad=True);out=torch.empty(64,device='cuda');grad=torch.empty_like(q)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            loss=residual(world,q);torch.autograd.grad(loss.sum(),q)
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(residual(world,q));grad.copy_(torch.autograd.grad(out.sum(),q)[0])
    with torch.no_grad():q[:,6]=.4
    graph.replay();torch.cuda.synchronize()
    assert out.max()>.01 and torch.isfinite(grad).all()
