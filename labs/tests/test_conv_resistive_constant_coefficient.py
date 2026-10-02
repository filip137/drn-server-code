"""Exact arithmetic and lifetime contracts for scoped float64 coefficient caches."""
from contextlib import nullcontext
import pytest
import torch
import torch.nn.functional as F

from model.resistive.interaction import ConvResistive
from model.variable.layer import LinearLayer
from model.variable.parameter import ConvWeight


def interaction(batch, shape, kernel, padding, stride, dilation, dtype, index=0):
    cin, height, width = shape
    cout = 5
    hout = (height + 2 * padding - dilation * (kernel[0] - 1) - 1) // stride + 1
    wout = (width + 2 * padding - dilation * (kernel[1] - 1) - 1) // stride + 1
    pre = LinearLayer(shape, batch_size=batch, device='cpu')
    post = LinearLayer((cout, hout, wout), batch_size=batch, device='cpu')
    pre._name, post._name = f'Layer_{index}', f'Layer_{index + 1}'
    pre.state = torch.randn(batch, *shape, dtype=dtype, requires_grad=True)
    post.state = torch.randn(batch, cout, hout, wout, dtype=dtype, requires_grad=True)
    weight = ConvWeight((cout, cin, *kernel), gain=1., device='cpu', clamp=True,
                        clamp_min=0., clamp_max=10.)
    weight.state = (torch.rand(cout, cin, *kernel, dtype=dtype) + .01).requires_grad_()
    return ConvResistive(pre, post, weight, padding, stride, dilation,
                        voltage_amp=4., current_amp=1.)


def original_coefficient(fn):
    weight, cout, _, kh, kw, height, width, hout, wout = fn._conv_geometry()
    ones = torch.ones_like(fn._layer_post.state).reshape(-1, cout, hout * wout)
    columns = torch.matmul(weight.view(cout, -1).t().unsqueeze(0), ones)
    return F.fold(columns, (height, width), (kh, kw),
                  padding=fn._P, stride=fn._S, dilation=fn._D)


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('index', [0, 2])
@pytest.mark.parametrize('shape,kernel,padding,stride,dilation', [
    ((2, 5, 7), (1, 1), 0, 1, 1),
    ((2, 5, 7), (3, 3), 1, 1, 1),
    ((3, 7, 6), (3, 2), 0, 1, 1),
    ((2, 6, 9), (2, 3), 1, 2, 1),
    ((3, 8, 9), (3, 2), 2, 1, 2),
    ((2, 6, 9), (2, 3), 0, 2, 2),
])
def test_float64_coefficients_and_all_weight_derivatives_match(
        batch, index, shape, kernel, padding, stride, dilation):
    torch.manual_seed(43)
    fn = interaction(batch, shape, kernel, padding, stride, dilation, torch.float64, index)
    fn._weight.state.requires_grad_(False)
    reference_scaled = fn._a_coef_layer_pre()
    with fn.cache_a_col2im():
        cached = fn.a_col2im()
        assert fn.a_col2im() is cached
        torch.testing.assert_close(cached, original_coefficient(fn), rtol=0, atol=0)
        torch.testing.assert_close(fn._a_coef_layer_pre(), reference_scaled, rtol=0, atol=0)
        fn._weight.state.requires_grad_(True)
        # A requested derivative must bypass the constant cache.
        expected = original_coefficient(fn)
        actual = fn.a_col2im()
    assert actual.shape == (batch, *shape) and actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Distinct upstream values exercise sample aggregation and every kernel
    # position, including padded, strided and dilated overlap patterns.
    probe = torch.randn_like(expected)
    reference_grad = torch.autograd.grad((expected.square() * probe).sum(), fn._weight.state)[0]
    gradient, pre_gradient, post_gradient = torch.autograd.grad(
        (actual.square() * probe).sum(),
        (fn._weight.state, fn._layer_pre.state, fn._layer_post.state), allow_unused=True)
    torch.testing.assert_close(gradient, reference_grad, rtol=0, atol=0)
    assert pre_gradient is None and post_gradient is None


@pytest.mark.parametrize('dtype,enabled', [(torch.float32, False), (torch.float32, True),
                                          (torch.float64, False)])
def test_original_path_is_exact_for_float32_or_disabled(dtype, enabled, monkeypatch):
    torch.manual_seed(47)
    fn = interaction(3, (2, 5, 7), (3, 3), 1, 1, 1, dtype)
    expected = original_coefficient(fn)
    calls = []
    original_matmul = torch.matmul
    def record_matmul(*args, **kwargs):
        calls.append(True)
        return original_matmul(*args, **kwargs)
    monkeypatch.setattr(torch, 'matmul', record_matmul)
    with fn.cache_a_col2im() if enabled else nullcontext():
        actual = fn.a_col2im()
    assert calls == [True]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(expected)
    expected_gradient = torch.autograd.grad((expected * probe).sum(), fn._weight.state)[0]
    actual_gradient = torch.autograd.grad((actual * probe).sum(), fn._weight.state)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_cache_is_disabled_outside_explicit_scope():
    fn = interaction(1, (2, 5, 7), (3, 3), 1, 1, 1, torch.float64)
    fn._weight.state.requires_grad_(False)
    assert fn._a_col2im_cache is None
    assert fn.a_col2im() is not fn.a_col2im()


def test_scoped_cache_is_exact_and_invalidates_weights_and_batch(monkeypatch):
    fn = interaction(3, (2, 5, 7), (3, 3), 1, 1, 1, torch.float64)
    fn._weight.state.requires_grad_(False)
    calls = []
    original = fn._a_col2im_uncached
    def compute():
        calls.append(True)
        return original()
    monkeypatch.setattr(fn, '_a_col2im_uncached', compute)
    with fn.cache_a_col2im():
        first = fn.a_col2im()
        assert fn.a_col2im() is first and len(calls) == 1
        torch.testing.assert_close(first, original_coefficient(fn), rtol=0, atol=0)
        fn._weight.state.add_(.1)
        second = fn.a_col2im()
        assert second is not first and len(calls) == 2
        torch.testing.assert_close(second, original_coefficient(fn), rtol=0, atol=0)
        fn._layer_pre.state = fn._layer_pre.state[:1]
        fn._layer_post.state = fn._layer_post.state[:1]
        third = fn.a_col2im()
        assert third.shape[0] == 1 and len(calls) == 3
        torch.testing.assert_close(third, original_coefficient(fn), rtol=0, atol=0)
    assert fn._a_col2im_cache is None
    fn.a_col2im()
    assert len(calls) == 4


def test_cache_bypasses_tracked_weight_gradients_and_restores_nested_scope():
    fn = interaction(3, (2, 5, 7), (3, 3), 1, 1, 1, torch.float64)
    with fn.cache_a_col2im():
        with torch.no_grad():
            cached = fn.a_col2im()
            assert fn.a_col2im() is cached
        actual = fn.a_col2im()
        reference = original_coefficient(fn)
        assert actual.requires_grad and not cached.requires_grad
        probe = torch.randn_like(actual)
        gradient = torch.autograd.grad((actual * probe).sum(), fn._weight.state)[0]
        expected = torch.autograd.grad((reference * probe).sum(), fn._weight.state)[0]
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
        outer = fn._a_col2im_cache
        with pytest.raises(RuntimeError, match='cache failure'):
            with fn.cache_a_col2im():
                assert fn._a_col2im_cache is not outer
                raise RuntimeError('cache failure')
        assert fn._a_col2im_cache is outer
        with torch.no_grad():
            assert fn.a_col2im() is cached
    assert fn._a_col2im_cache is None


def test_scoped_cache_does_not_change_float32():
    fn = interaction(3, (2, 5, 7), (3, 3), 1, 1, 1, torch.float32)
    fn._weight.state.requires_grad_(False)
    with fn.cache_a_col2im():
        first, second = fn.a_col2im(), fn.a_col2im()
        assert first is not second and fn._a_col2im_cache == {}
        torch.testing.assert_close(first, second, rtol=0, atol=0)
