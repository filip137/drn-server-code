# ReLU to DRN Initialization

Use this when initializing a positive-conductance DRN from a trained signed
ReLU teacher.

## Architecture

The teacher should be an ordinary signed-weight ReLU MLP:

```text
784 -> 100 -> 10
```

The DRN carries signs physically:

```text
1568 -> 200 -> 10
input:  [x, -x]
hidden: [h_pos, h_neg]
output: 10 units
```

Do not double the teacher input or the DRN output.

## Conductance Normalization

Normalize signed teacher weights by maximum absolute value, then split signs.
Do not min-max shift signed weights into the conductance range.

```python
G_MAX = 180e-6
eps = 1e-12

s1 = G_MAX / W1.abs().max().clamp_min(eps)
s2 = G_MAX / W2.abs().max().clamp_min(eps)

W1_scaled = s1 * W1
W2_scaled = s2 * W2

W1_pos = W1_scaled.clamp_min(0.0)
W1_neg = (-W1_scaled).clamp_min(0.0)
W2_pos = W2_scaled.clamp_min(0.0)
W2_neg = (-W2_scaled).clamp_min(0.0)
```

This preserves the signed effective weights:

```text
G_pos - G_neg = s * W
0 <= G_pos, G_neg <= 180e-6
```

## Packing

For `W1` shaped `(784, 100)`:

```python
G_in = torch.zeros(1568, 200, dtype=W1.dtype, device=W1.device)
G_in[:784, :100] = W1_pos
G_in[784:, :100] = W1_neg
G_in[:784, 100:] = W1_neg
G_in[784:, 100:] = W1_pos
```

For `W2` shaped `(100, 10)`:

```python
G_out = torch.cat((W2_pos, W2_neg), dim=0)
```

## Bias Scaling

Scale hidden bias in the same gauge as the first layer. Do not normalize bias by
its own max value.

```python
b_hidden = torch.cat((s1 * b1, -s1 * b1), dim=0)
```

If the DRN input path multiplies `[x, -x]` by `input_gain`, include it:

```python
b_hidden = torch.cat((input_gain * s1 * b1, -input_gain * s1 * b1), dim=0)
```

The output bias can be copied directly when the student has an explicit output
bias:

```python
b_output = b2
```

## Checks

Before training or evaluation, verify:

```text
teacher output shape = (batch, 10)
student output shape = (batch, 10)
all conductances are finite
0 <= conductances <= 180e-6
hidden bias uses the first-layer weight scale
```
