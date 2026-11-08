"""Unit tests for cuRobo workarounds in _curobo_internals.py."""
import torch


def test_patches_are_installed_and_the_guard_names_a_changed_function(monkeypatch):
    """``import cutamp`` installed the patches: a URDF link with no inertial weighs nothing (test_robot_com.py checks
    the values), and one ``wp.Stream`` serves every launch on a CUDA stream. The source guard rejects a cuRobo whose
    pinned functions differ from v0.9.0's, naming the function."""
    import pytest
    import warp as wp
    from curobo._src.robot.parser.parser_urdf import UrdfRobotParser

    import cutamp._curobo_internals as ci

    assert hasattr(UrdfRobotParser.get_link_parameters, "__wrapped__")
    if torch.cuda.is_available():
        wp.init()  # cuRobo initializes warp before its first launch
        stream = torch.cuda.current_stream()
        assert wp.stream_from_torch(stream) is wp.stream_from_torch(stream)
    path = "curobo._src.util.warp:get_warp_device_stream"
    monkeypatch.setitem(ci._PINNED_SOURCES, path, "0" * 12)
    with pytest.raises(RuntimeError, match="get_warp_device_stream"):
        ci._check_curobo_sources()


def test_kernel_disk_cache_reuses_a_cubin_until_an_input_changes(tmp_path, monkeypatch):
    """A second compile of the same kernel loads the stored cubin and leaves its file alone; an edited header or a
    changed flag compiles afresh under a new key. An entry whose kernel doesn't load is compiled again and replaced, a
    header edited after the process took its digest isn't stored, and an unreadable file under an include path
    compiles uncached. The loaded cubin runs."""
    import pickle

    import cuda.core
    import pytest

    import cutamp._curobo_internals as ci

    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA device.")
    device = cuda.core.Device()
    device.set_current()
    cache = tmp_path / "cache"
    monkeypatch.setattr(ci, "_KERNEL_CACHE_DIR", str(cache))
    inc = tmp_path / "inc"
    inc.mkdir()
    src = '#include "scale.cuh"\nextern "C" __global__ void scale(float* x) { x[threadIdx.x] *= SCALE; }\n'

    def compile_scale(factor, fresh_digest=True, **flags):
        (inc / "scale.cuh").write_text(f"#define SCALE {factor}f\n")
        if fresh_digest:
            ci._tree_digest.cache_clear()
        arch = "sm_%d%d" % torch.cuda.get_device_capability()
        options = cuda.core.ProgramOptions(arch=arch, include_path=[str(inc)], **flags)
        return cuda.core.Program(src, code_type="c++", options=options).compile("cubin", name_expressions=("scale",))

    def entries():
        return sorted(cache.glob("*.pkl"))

    compile_scale(2.0)
    (entry,) = entries()
    inode = entry.stat().st_ino
    mod = compile_scale(2.0)
    assert entries() == [entry] and entry.stat().st_ino == inode  # a hit rewrites nothing
    compile_scale(3.0)
    compile_scale(2.0, ftz=True)
    assert len(entries()) == 3  # the header and every flag are in the key
    entry.write_bytes(pickle.dumps((b"not a cubin", {})))  # unpickles; only loading its kernel fails
    compile_scale(2.0)
    assert len(entries()) == 3 and entry.stat().st_ino != inode  # compiled again and replaced
    compile_scale(4.0, fresh_digest=False, fma=False)  # the header changed under the process's digest of it
    assert len(entries()) == 3
    (inc / ".#scale.cuh").symlink_to(tmp_path / "gone")  # an editor's lock file: a dangling link
    compile_scale(5.0)
    assert len(entries()) == 3

    x = torch.ones(4, device="cuda")
    torch.cuda.synchronize()
    stream = device.create_stream()
    cuda.core.launch(stream, cuda.core.LaunchConfig(grid=1, block=4), mod.get_kernel("scale"), x.data_ptr())
    stream.sync()
    assert x.tolist() == [2.0] * 4
