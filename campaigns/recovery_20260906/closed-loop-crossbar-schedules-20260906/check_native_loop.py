"""Compare the adapter with the native gradient/update loop on this GPU."""
import json
from itertools import islice
import sys
import torch
from run_recovery import HERE, Crossbar, schedule_factor, write_json

sys.path.insert(0, str(HERE/'crossbar_code'))
torch.set_num_threads(1)
assert torch.cuda.is_available()
plan = json.loads((HERE/'plan.json').read_text())
rows = []
for condition in ('healthy', 'faulted'):
    lr = plan['crossbar']['learning_rates'][condition]
    left, right = [Crossbar(plan, condition, lr, 128) for _ in range(2)]
    for epoch in (1, 2):
        scale = schedule_factor(plan['schedules']['exponential'], epoch)
        left.open_optimizer.learning_rate_scale = scale
        right.open_optimizer.learning_rate_scale = scale
        pairs = zip(islice(left.data.train, 256), islice(right.data.train, 256), strict=True)
        for (x,y),(xr,yr) in pairs:
            assert torch.equal(x,xr) and torch.equal(y,yr)
            x,y = x.to(left.device),y.to(left.device)
            gradient,loss = left.gradient(x,y)
            q = right.port.apparent.to(device=right.device).requires_grad_(True)
            logits = right.engine.standard_crossbar_logits(x,q,right.layout,digital_scales=right.origin.current.digital_scales)
            native_loss = right.engine._adam_objective_loss(objective='teacher_kl',logits=logits,labels=y,inputs=x,teacher=right.teacher)
            native_gradient = torch.autograd.grad(native_loss,q,only_inputs=True)[0]
            assert torch.equal(gradient,native_gradient) and loss == float(native_loss.item())
            left.open_step(gradient)
            right.open_optimizer.step(native_gradient,right.port)
            assert torch.equal(left.apparent(),right.apparent())
            assert torch.equal(left.persistent(),right.persistent())
            assert torch.equal(left.open_optimizer.generator.get_state(),right.open_optimizer.generator.get_state())
    rows.append(dict(condition=condition,minibatches=512,gradient_and_plant_bitwise_equal=True,
                     pulses=int(left.open_optimizer.pulse_count.sum())))
write_json(HERE/'smoke/native_loop_parity.json',dict(status='passed',cases=rows))
print(json.dumps(rows))
