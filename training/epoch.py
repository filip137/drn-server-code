import os
import sys
#from tkinter.constants import FALSE
import torch
from pathlib import Path  # add near the top, next to the existing imports

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.resistive.interaction import BasePoolResistive, ConvResistive, DenseResistive
import torch.nn.functional as F



class Epoch:
    """
    Class used to process a dataset with a network once (one 'epoch') to change the internal state of the network and/or compute statistics.
    Two important subclasses of Epoch are Evaluator and Trainer.

    Attributes
    ----------
    _stats (list of list of Statistic): the list of all lists of statistics to be computed over the dataset

    Methods
    -------
    add_statistic(stat, list_idx)
        Adds a statistic to the collection of index list_idx
    dataset_size()
        Returns the size of the dataset
    _reset()
        Sets all the statistics to zero
    _do_measurements(list_idx)
        Do the measurements for each of the statistics in list of index list_idx
    """

    def __init__(self, num_lists):
        """Creates an instance of Epoch

        Args:
            num_lists (int): the number of lists of statistics
        """

        self._stats = []
        for _ in range(num_lists): self._stats.append([])

    def add_statistic(self, stat, list_idx=0):
        """Adds a statistic to the list of statistics

        Args:
            stat (Statistic): the statistic to be added
            list_idx (int, optional): the index of the list in which we add the statistic. Default: 0
        """

        self._stats[list_idx].append(stat)

    def dataset_size(self):
        """Returns the size of the dataset"""
        return len(self._dataloader.dataset)

    def _all_stats(self):
        """Returns the list of all stats, both from the 'eval' and 'train' collections"""
        return [stat for list_stats in self._stats for stat in list_stats]

    def _reset(self):
        """Sets all the statistics to zero"""

        for stat in self._all_stats(): stat.reset()

    def _do_measurements(self, list_idx=0):
        """Do measurements in all stats of the collection of index list_idx

        Args:
            list_idx (int, optional): the index of the list where we measure all the statistic. Default: 0
        """

        for stat in self._stats[list_idx]: stat.do_measurement()

    def __str__(self):
        list_of_strings = [str(stat) for stat in self._all_stats() if stat.display]
        string = ', '.join(list_of_strings)
                
        return string



class BetaSize(Epoch):
    """Measure per-layer displacement between free and nudged equilibria (no updates)."""

    def __init__(self, network, cost_fn, params, dataloader, differentiator, energy_minimizer, max_batches=None, eps=1e-12):
        Epoch.__init__(self, 2)
        self._network = network
        self._params = params + cost_fn.params()
        self._cost_fn = cost_fn
        self._dataloader = dataloader
        self._differentiator = differentiator
        self._energy_minimizer = energy_minimizer
        self._max_batches = max_batches
        self._eps = eps
        self._ratios_per_batch, self._free_rms_per_batch, self._delta_rms_per_batch = [], [], []
        self._layer_names = [getattr(l, "name", f"layer{i}") for i, l in enumerate(self._network.layers())]

    @torch.no_grad()
    def _measure_displacement(self, layers_free, layers_first):
        free_rms, delta_rms, ratios = [], [], []
        for z0, z1 in zip(layers_free.values(), layers_first.values()):          
            #fr = self._rms_tensor(z0)
            #dr = self._rms_tensor(z1 - z0)
            avg_disp = (z0 - z1).abs().mean()
            free_avg = z0.abs().mean()
            free_rms.append(free_avg)
            delta_rms.append(avg_disp)
            ratios.append(avg_disp / (free_avg + self._eps))
        return torch.stack(free_rms).detach(), torch.stack(delta_rms).detach(), torch.stack(ratios).detach()

    @torch.no_grad()
    def run(self, verbose=False):
        self._reset(); self._ratios_per_batch.clear(); self._free_rms_per_batch.clear(); self._delta_rms_per_batch.clear()
        b = 0
        for batch in self._dataloader:
            x, y = batch[:2]
            self._network.set_input(x, reset=True)
            self._energy_minimizer.compute_equilibrium()  # free
            self._cost_fn.set_target(y)
            layers_free, layers_first = self._differentiator.return_free_and_nudged_states()  # nudged (weights unchanged)
            free_rms, delta_rms, ratios = self._measure_displacement(layers_free, layers_first)
            self._free_rms_per_batch.append(free_rms); self._delta_rms_per_batch.append(delta_rms); self._ratios_per_batch.append(ratios)
            if verbose:
                med = ratios.median().item(); sys.stdout.write(f"\rBetaSize batch {b+1}: median ratio = {med:.4f}"); sys.stdout.flush()
            b += 1
            if self._max_batches is not None and b >= self._max_batches: break
        if verbose: sys.stdout.write("\n")

    def ratios_tensor(self):
        if not self._ratios_per_batch: return torch.empty((0, len(self._layer_names)))
        return torch.stack(self._ratios_per_batch, dim=0)


    def _rms_tensor(self, t: dict) -> torch.Tensor:
        """Root-mean-square over all elements."""
        values = t
        return torch.sqrt(torch.mean(values.float() * values.float()))
    def summary(self):
        R = self.ratios_tensor()
        if R.numel() == 0: return {"layers": self._layer_names, "per_layer_median": [], "per_layer_mean": []}
        per_layer_median = R.median(dim=0).values
        per_layer_mean = R.mean(dim=0)
        return {
            "layers": self._layer_names,
            "per_layer_median": per_layer_median.cpu().tolist(),
            "per_layer_mean":  per_layer_mean.cpu().tolist(),
            "overall_median":  R.median().item(),
            "overall_mean":    R.mean().item()
        }


    

class Evaluator(Epoch):
    """
    Class used to evaluate a network on a dataset

    Attributes
    ----------
    _network (SumSeparableFunction): the model to evaluate
    _dataloader (Dataloader): the dataset on which to evaluate the model
    energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference

    idx (tensor of int): vector of indices of the data examples in the current mini-batch

    Methods
    -------
    run(verbose)
        Evaluates the network over the dataset
    """

    def __init__(self, network, cost_fn, dataloader, energy_minimizer):
        """Initializes an instance of Evaluator

        Args:
            network (Network): the model to evaluate
            cost_fn (CostFunction): the cost function to optimize
            dataloader (Dataloader): the dataset on which to evaluate the model. An IndexedDataset that loads data in the form of triplets (x, y, idx)
            energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference
        """

        Epoch.__init__(self, 1)

        self._network = network
        self._cost_fn = cost_fn
        self._dataloader = dataloader

        self._energy_minimizer = energy_minimizer

    @property
    def idx(self):
        """Gets the indices of the examples in the last mini batch processed"""
        return self._idx

    def run(self, verbose=False):
        """Evaluate the model over the dataset.

        Args:
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after processing the entire dataset. Default: False.
        """

        self._reset()  # sets all the statistics to zero

        for x, y, idx in self._dataloader:

            # Inference (free phase relaxation)
            self._network.set_input(x, reset=True)
            self._energy_minimizer.compute_equilibrium()

            # Measure statistics
            self._idx = idx
            self._cost_fn.set_target(y)
            self._do_measurements()

            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write(str(self))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.write(str(self))
            sys.stdout.write('\n')

    def __str__(self):
        return 'TEST  -- ' + Epoch.__str__(self)



class Trainer(Epoch):
    """
    Class used to train a network on a dataset

    Attributes
    ----------
    _network (SumSeparableFunction): the network to train
    _dataloader (Dataloader): the dataset on which to train the network
    _differentiator (GradientEstimator): the method used to train the network
    energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference

    _stats (list of Statistic): _stats[0] is the list of statistics to measure after inference (evaluation), and _stats[1] is the list of statistics to measure after computing the gradient (training)

    Methods
    -------
    run(verbose)
        Train the network for one epoch over the dataset
    """

    def __init__(self, network, cost_fn, params, dataloader, differentiator, optimizer, energy_minimizer):
        """Initializes an instance of Trainer

        Args:
            network (Network): the network to train
            cost_fn (CostFunction): the cost function to optimize
            dataloader (Dataloader): the dataset on which to train the network
            differentiator (GradientEstimator): either EquilibriumProp or Backprop
            optimizer (str): the optimizer used to optimize.
            energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference
        """


        Epoch.__init__(self, 2)

        self._network = network
        self._params = params + cost_fn.params()  # FIXME
        self._cost_fn = cost_fn
        self._dataloader = dataloader
        self._differentiator = differentiator
        self._optimizer = optimizer
        self._energy_minimizer = energy_minimizer

    def track_gradient_and_update_sizes(self, grads, verbose=False):
        """
        Track the size of gradients and parameter updates.
        
        Args:
            grads (list of Tensor): the computed gradients
            verbose (bool): if True, print detailed statistics
            
        Returns:
            dict: statistics about gradients and updates
        """
        
        # Store parameters before update
        params_before = [param.state.clone() for param in self._params]
        
        # Calculate gradient statistics
        gradient_stats = {}
        total_grad_norm = 0.0
        total_grad_mean = 0.0
        
        for i, (param, grad) in enumerate(zip(self._params, grads)):
            param_name = param.name
            
            # Gradient statistics
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            grad_max = grad.max().item()
            grad_min = grad.min().item()
            
            gradient_stats[param_name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max,
                'min': grad_min,
                'shape': grad.shape
            }
            
            total_grad_norm += grad_norm
            total_grad_mean += grad_mean
            
            if verbose:
                print(f"  {param_name}: grad_norm={grad_norm:.6f}, grad_mean={grad_mean:.6f}, "
                      f"grad_std={grad_std:.6f}, grad_range=[{grad_min:.6f}, {grad_max:.6f}]")
        
        # Perform the optimizer step
        self._optimizer.step()
        
        # Calculate update statistics
        update_stats = {}
        total_update_norm = 0.0
        total_update_mean = 0.0
        
        for i, (param, param_before) in enumerate(zip(self._params, params_before)):
            param_name = param.name
            
            # Calculate the update (difference after optimizer.step())
            update = param.state - param_before
            
            # Update statistics
            update_norm = update.norm().item()
            update_mean = update.mean().item()
            update_std = update.std().item()
            update_max = update.max().item()
            update_min = update.min().item()
            
            update_stats[param_name] = {
                'norm': update_norm,
                'mean': update_mean,
                'std': update_std,
                'max': update_max,
                'min': update_min,
                'shape': update.shape
            }
            
            total_update_norm += update_norm
            total_update_mean += update_mean
            
            if verbose:
                print(f"  {param_name}: update_norm={update_norm:.6f}, update_mean={update_mean:.6f}, "
                      f"update_std={update_std:.6f}, update_range=[{update_min:.6f}, {update_max:.6f}]")
        
        # Summary statistics
        summary_stats = {
            'total_grad_norm': total_grad_norm,
            'total_grad_mean': total_grad_mean,
            'total_update_norm': total_update_norm,
            'total_update_mean': total_update_mean,
            'gradient_stats': gradient_stats,
            'update_stats': update_stats,
            'num_parameters': len(self._params)
        }
        
        if verbose:
            print(f"\nSummary:")
            print(f"  Total gradient norm: {total_grad_norm:.6f}")
            print(f"  Total gradient mean: {total_grad_mean:.6f}")
            print(f"  Total update norm: {total_update_norm:.6f}")
            print(f"  Total update mean: {total_update_mean:.6f}")
            print(f"  Number of parameters: {len(self._params)}")
        
        return summary_stats

    def run(self, verbose=True):
        """Train the model for one epoch over the dataset.

        Args:
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after every epoch. Default: True.
        """

        self._reset()  # sets all the statistics to zero
        debug = False
        diffs = {}
        cosis = {}
        matrixnorm_torch = {}
        matrixnorm_analytic = {}
        self._last_b_components = {}
        for x, y in self._dataloader:

            # inference (free phase relaxation)
            self._network.set_input(x, reset=True)  # we set the input, and we let the state of the network where it was at the end of the previous batch

            self._energy_minimizer.compute_equilibrium()  # we let the network settle to equilibrium (free state)

            torch_grads_free = {}
            if debug:
                for interaction in self._network._function._interactions:
                    if (isinstance(interaction, DenseResistive) or isinstance(interaction, ConvResistive)):
                        weight = interaction.params()[0]
                        torch_grads_free[weight.name] = interaction._grad(weight, mean=True)

            self._cost_fn.set_target(y)  # we present the correct (desired) output
            self._do_measurements(0)  # we measure the statistics of the free state (energy value, cost value, error value, ...)

            # training step
            grads = self._differentiator.compute_gradient()  # compute the parameter gradients

            for param, grad in zip(self._params, grads): param.state.grad = grad  # Set the gradients of the parameters
            self._do_measurements(1)  # measure the statistics of training
            
            if debug:
                for interaction in self._network._function._interactions:
                    if (isinstance(interaction, BasePoolResistive)):
                        layer_pre = interaction._layer_pre
                        layer_post = interaction._layer_post
                    #if isinstance(interaction, DenseResistive) or isinstance(interaction, ConvResistive):
                    elif (isinstance(interaction, DenseResistive) or isinstance(interaction, ConvResistive)):
                        weight = interaction.params()[0]
                        layer_pre = interaction._layer_pre
                        layer_post = interaction._layer_post
                        pre_grad = interaction.grad_layer_fn(layer_pre)()
                        post_grad = interaction.grad_layer_fn(layer_post)()
                        key = weight.name
                        torch_grad = torch_grads_free.get(weight.name, interaction._grad(weight, mean=True))
                        torch_grad_norm = torch.norm(torch_grad).item() / torch_grad.numel()**0.5
                        analytic = interaction._grad_weight()
                        analytic_norm  = torch.norm(analytic).item() / analytic.numel()**0.5
                        a = analytic.reshape(-1)
                        b = torch_grad.reshape(-1)
                        cosine = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()

                        diff = torch.norm(analytic - torch_grad).item()
                        diffs.setdefault(weight.name, []).append(diff)
                        cosis.setdefault(weight.name, []).append(cosine)
                        matrixnorm_torch.setdefault(weight.name, []).append(torch_grad_norm)
                        matrixnorm_analytic.setdefault(weight.name, []).append(analytic_norm)

                # Collect b/a term components per layer for debugging/inspection
                with torch.no_grad():
                    self._last_b_components = {}
                    energy_fn = getattr(self._network, "_function", None)
                    eps = 1e-12
                    debug_diode = os.environ.get("DRN_DEBUG_DIODE")
                    def _print_non_finite(label, tensor):
                        nan_count = torch.isnan(tensor).sum().item()
                        inf_count = torch.isinf(tensor).sum().item()
                        print(
                            "[a/b debug] "
                            f"{label} nan={nan_count} inf={inf_count} "
                            f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                        )
                    if energy_fn is not None:
                        for layer in energy_fn.layers():
                            a_total = energy_fn.a_coef_fn(layer)()
                            comps = energy_fn.b_coef_components(layer)
                            if not comps:
                                continue
                            name = getattr(layer, "name", f"layer_{id(layer)}")
                            if debug_diode and not torch.isfinite(a_total).all():
                                _print_non_finite(f"{name} a_total", a_total)
                                for comp in energy_fn.a_coef_components(layer):
                                    val = comp["value"]
                                    if not torch.isfinite(val).all():
                                        _print_non_finite(f"{name} a {comp['interaction']}", val)
                            summaries = []
                            for comp in comps:
                                val = comp["value"]
                                if debug_diode and not torch.isfinite(val).all():
                                    _print_non_finite(f"{name} b {comp['interaction']}", val)
                                ratio = val / (a_total + eps)
                                summaries.append({
                                    "interaction": comp["interaction"],
                                    "mean": ratio.mean().item(),
                                    "min": ratio.min().item(),
                                    "max": ratio.max().item(),
                                })
                            self._last_b_components[name] = summaries
                        if verbose and self._last_b_components:
                            print("\n[b/a components]")
                            for lname, comps in self._last_b_components.items():
                                print(f"  {lname}:")
                                for c in comps:
                                    print(f"    {c['interaction']}: mean={c['mean']:.4e}, min={c['min']:.4e}, max={c['max']:.4e}")


            # Track gradient/update sizes and apply optimizer step
            stats = self.track_gradient_and_update_sizes(grads, verbose=False)
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write(str(self))
                sys.stdout.flush()
            

            for param in self._params: param.clamp_()  # clamp the parameters' states in their range of permissible values, if adequate



    def __str__(self):
        return 'TRAIN -- ' + Epoch.__str__(self)
