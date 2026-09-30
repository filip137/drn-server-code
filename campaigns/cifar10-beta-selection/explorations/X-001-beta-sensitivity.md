# X-001 — Does CIFAR tolerate large beta?

Filip's starting observation is that MNIST can learn well with relatively large
beta. CIFAR has a different blockwise architecture and digital pooling/BN
boundaries, so that observation motivates a test rather than establishing a rule.

The [CIFAR pilot](../../pilots/cifar-block-beta-20260929.md) matched relative
physical-output displacement4 for three-conv blocks and1 for the two-conv block.
Its initial cosine results are weak, especially at the three-conv outputs.
This retrospective observation motivates the prospective smaller-beta sweep.

Possible explanations include nonlinear response at large beta, insufficient
finite settling, and limited float32 resolution of weak early-layer responses.
Lowering beta at fixed T/K isolates the beta dependence but may expose the latter
limitations. A stable energy-difference calculation is already validated.

Gradient alignment does not establish training quality. Follow with matched
10-epoch runs at selected beta values, with explicit treatment of digital
bridges, BN/gains, the readout and the blockwise EqProp gradient estimator.
The existing native CIFAR trainer is BPTT and ignores beta; relabelling a BPTT
run would not test this idea.

Focused follow-up: exp-001 shows a beta-dependent high-alignment region, with an early-layer deterioration at very small B. Filip asks whether expressing this region as output RMS produces a common rule across blocks. Exp-003 uses separately chosen B samples at comparable relative RMS, and also checks absolute RMS as a competing scaling description. A common usable interval and identical curve shapes are distinct claims.
