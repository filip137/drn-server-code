"""The LR intervention preserves Adam history and future schedule scaling."""
import copy
import unittest
import torch
from experiments.screen_cifar_l8_conv_lr import scale_convolution_rates


class ConvLRScreenTest(unittest.TestCase):
    def test_warm_adam_update_and_next_epoch(self):
        def build():
            params = [torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True) for _ in range(3)]
            opt = torch.optim.Adam([{'params':[p], 'name':n, 'lr':lr}
                                   for p,n,lr in zip(params,['conv','head','bn'],[.02,.01,.03])], foreach=False)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: .9**e)
            return params, opt, scheduler
        original, opt, sched = build()
        for _ in range(3):
            for p in original:p.grad = torch.tensor([.4, -.1], dtype=torch.float64)
            opt.step(); sched.step()
        arms = []
        for factor in [1., 3.]:
            params, optimizer, scheduler = build()
            with torch.no_grad():
                for p, q in zip(params, original):p.copy_(q)
            optimizer.load_state_dict(copy.deepcopy(opt.state_dict()))
            scheduler.load_state_dict(copy.deepcopy(sched.state_dict()))
            moments = copy.deepcopy(optimizer.state_dict()['state'])
            scale_convolution_rates(optimizer, scheduler, {'conv'}, factor)
            for key, state in optimizer.state_dict()['state'].items():
                for field, tensor in state.items():self.assertTrue(torch.equal(tensor, moments[key][field]))
            for p in params:p.grad = torch.tensor([-.2, .3], dtype=torch.float64)
            optimizer.step(); scheduler.step()
            arms.append((params, optimizer, scheduler))
        control, changed = arms
        torch.testing.assert_close(changed[0][0]-original[0], 3*(control[0][0]-original[0]), atol=1e-14, rtol=1e-12)
        for i in [1,2]:self.assertTrue(torch.equal(control[0][i], changed[0][i]))
        self.assertAlmostEqual(changed[2].get_last_lr()[0], 3*control[2].get_last_lr()[0])
        self.assertEqual(changed[2].get_last_lr()[1:], control[2].get_last_lr()[1:])
        self.assertEqual(changed[2].last_epoch, control[2].last_epoch)


    def test_reduced_head_preserves_other_updates_and_schedule(self):
        def build():
            params = [torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True) for _ in range(3)]
            opt = torch.optim.Adam([{'params':[p], 'name':n, 'lr':lr}
                                   for p,n,lr in zip(params,['conv','head','bn'],[.02,.01,.03])], foreach=False)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: .9**e)
            return params, opt, scheduler
        original, opt, sched = build()
        for _ in range(3):
            for p in original:p.grad = torch.tensor([.4, -.1], dtype=torch.float64)
            opt.step(); sched.step()
        arms = []
        for factor in [1., 1/27]:
            params, optimizer, scheduler = build()
            with torch.no_grad():
                for p, q in zip(params, original):p.copy_(q)
            optimizer.load_state_dict(copy.deepcopy(opt.state_dict()))
            scheduler.load_state_dict(copy.deepcopy(sched.state_dict()))
            moments = copy.deepcopy(optimizer.state_dict()['state'])
            scale_convolution_rates(optimizer, scheduler, {'head'}, factor)
            for key, state in optimizer.state_dict()['state'].items():
                for field, tensor in state.items():self.assertTrue(torch.equal(tensor, moments[key][field]))
            for p in params:p.grad = torch.tensor([-.2, .3], dtype=torch.float64)
            optimizer.step(); scheduler.step()
            arms.append((params, optimizer, scheduler))
        control, changed = arms
        torch.testing.assert_close(changed[0][1]-original[1], (control[0][1]-original[1])/27, atol=1e-14, rtol=1e-12)
        for i in [0,2]:self.assertTrue(torch.equal(control[0][i], changed[0][i]))
        self.assertAlmostEqual(changed[2].get_last_lr()[1], control[2].get_last_lr()[1]/27)
        self.assertEqual([changed[2].get_last_lr()[i] for i in [0,2]], [control[2].get_last_lr()[i] for i in [0,2]])
        self.assertEqual(changed[2].last_epoch, control[2].last_epoch)


if __name__ == '__main__':unittest.main()
