
import copy
import torch
import torch.nn as nn
import numpy as np
import scipy.optimize as optim
from tqdm import tqdm
from itertools import count
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantModel, BaseQuantization
from trailmet.algorithms.quantize.quantize import Conv2dFunctor, LinearFunctor
from trailmet.algorithms.quantize.methods import LearnedStepSizeQuantization, FixedClipValueQuantization
from trailmet.algorithms.quantize.qmodel import ParameterModuleWrapper, ActivationModuleWrapper
from trailmet.algorithms.quantize.qmodel import QuantModule, BaseQuantBlock


class QuantModel(BaseQuantModel):
    def __init__(self, model: nn.Module, weight_quant_params: dict, act_quant_params: dict, fold_bn=True):
        super().__init__(model, weight_quant_params, act_quant_params, fold_bn) 
        self.quantizers = []
        for module in self.model.modules():
            if isinstance(module, QuantModule):
                self.quantizers.append(module.weight_quantizer)
                self.quantizers.append(module.act_quantizer)
            elif isinstance(module, BaseQuantBlock):
                self.quantizers.append(module.act_quantizer)

    def get_alphas_np(self):
        alphas = []
        for quantizer in self.quantizers:
            alphas.append(quantizer.alpha)
        return torch.tensor(alphas).numpy()
    
    def set_alphas_np(self, alphas: np.ndarray):
        for i, quantizer in  enumerate(self.quantizers):
            quantizer.set_params_from_alpha(torch.tensor(alphas[i]))
        

class LAPQ(BaseQuantization):
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(LAPQ, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['test']
        self.kwargs = kwargs
        self.w_bits = kwargs.get('W_BITS', 8)
        self.a_bits = kwargs.get('A_BITS', 8)
        self.calib_batches = kwargs.get('CALIB_BATCHES', 16)
        self.act_quant = kwargs.get('ACT_QUANT', True)
        self.test_before_calibration = kwargs.get('DRY_RUN', True)
        self.maxiter = kwargs.get('MAX_ITER', 1)
        self.maxfev = kwargs.get('MAX_FEV', 1)
        self.verbose = kwargs.get('VERBOSE', True)
        self.print_freq = kwargs.get('PRINT_FREQ', 20)
        self.gpu_id = kwargs.get('GPU_ID', 0)
        self.seed = kwargs.get('SEED', 42)
        seed_everything(self.seed)
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        if self.verbose:
            print("==> Using seed: {} and device: cuda:{}".format(self.seed, self.gpu_id))
        self.calib_data = self.get_calib_samples(self.train_loader, 64*self.calib_batches)
        self.eval_count = count(0)
        self.min_loss = 1e6

    def compress_model(self):
        self.model.to(self.device)
        self.model.eval()
        # self.search_absorbe_bn(self.model)
        weight_quant_params = {
            'n_bits': self.w_bits,
            'bcorr': True,
            'method': 'lp_norm',
            'p_val': 2.0,
        }
        act_quant_params = {
            'n_bits': self.w_bits,
            'bcorr': True,
            'method': 'lp_norm',
            'p_val': 2.0,
        }

        p_vals = np.linspace(2,4,10)
        losses = []
        pbar = tqdm(p_vals, total=len(p_vals))
        for p in pbar:
            weight_quant_params['p_val'] = p
            act_quant_params['p_val'] = p
            qnn = QuantModel(self.model, weight_quant_params, act_quant_params)
            loss = self.evaluate_loss(qnn, self.device)
            losses.append(loss.item())
            pbar.set_postfix(p_val=p, loss=loss.item())
            del qnn
        # using quadratic interpolation to approximate the optimal quantization step size ∆p∗
        z = np.polyfit(p_vals, losses, 2)
        y = np.poly1d(z)
        p_intr = y.deriv().roots[0]
        print("==> using p val : {:.2f}  with lp-loss : {:.2f}".format(p_intr, min(losses)))

        weight_quant_params['p_val'] = p_intr
        act_quant_params['p_val'] = p_intr

        self.qnn = QuantModel(self.model, weight_quant_params, act_quant_params)
        lp_acc1, lp_acc5 = self.test(self.qnn, self.test_loader, device=self.device)
        if self.verbose:
            print('==> Quantization (W{}A{}) accuracy before Optimization: {:.4f} | {:.4f}'.format(
                self.w_bits, self.a_bits, lp_acc1, lp_acc5))
            print("==> Starting Powell Optimization")

        init_alphas = self.qnn.get_alphas_np()
        min_method = "Powell"
        min_options = {
            'maxiter' : self.maxiter,
            'maxfev' : self.maxfev
        }
        count_iter = count(0)
        def local_search_callback(x):
            it = next(count_iter)
            self.qnn.set_alphas_np(x)
            loss = self.evaluate_loss(self.qnn.model, self.device)
            if self.verbose:
                print('\n==> Loss at end of iter [{}] : {:.4f}\n'.format(it, loss.item()))

        self.pbar = tqdm(total=min(self.maxiter, self.maxfev))
        res = optim.minimize(
            lambda scales: self.evaluate_calibration(scales, self.qnn, self.device), init_alphas,
            method=min_method, options=min_options, callback=local_search_callback
        )
        self.pbar.close()
        alphas = res.x
        if self.verbose:
            print('==> Layer-wise Scales :\n', alphas)
        self.qnn.set_alphas_np(alphas)
        print('==> Full quantization (W{}A{}) accuracy: {}'.format(
            self.w_bits, self.a_bits, 
            self.test(self.qnn, self.test_loader, device=self.device)))
        return self.qnn


    def evaluate_calibration(self, alphas: np.ndarray, qmodel: QuantModel, device):
        eval_count = next(self.eval_count)
        qmodel.set_alphas_np(alphas)
        loss = self.evaluate_loss(qmodel, device).item()
        if loss < self.min_loss:
            self.min_loss = loss
        self.pbar.set_postfix(curr_loss=loss, min_loss=self.min_loss)
        self.pbar.update(1)
        return loss

    def evaluate_loss(self, model: nn.Module, device):
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.eval()
        with torch.no_grad():
            if not hasattr(self, 'cal_set'):
                self.cal_set = []
                for i, (images, target) in enumerate(self.train_loader):
                    if i>=self.calib_batches:             # TODO: make this robust for variable batch size
                        break
                    images = images.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    self.cal_set.append((images, target))
            res = torch.tensor([0.]).to(device)
            for i in range(len(self.cal_set)):
                images, target = self.cal_set[i]
                output = model(images)
                loss = criterion(output, target)
                res += loss
            return res / len(self.cal_set)        





# class QuantModel_v1:
#     def __init__(self, model, args, quantizable_layers, optimizer_bridge=None):
#         self.model = model
#         self.args = args
#         self.bit_weights = args['bit_weights']
#         self.bit_act = args['bit_act']
#         self.post_relu = True
        
#         self.replacement_factory = {
#             nn.ReLU: ActivationModuleWrapper,
#             nn.ReLU6: ActivationModuleWrapper,
#             nn.Conv2d: ParameterModuleWrapper 
#             }
#         self.functor_map = {
#             nn.Conv2d: Conv2dFunctor, 
#             nn.Linear: LinearFunctor, 
#             }
#         self.optimizer_bridge = optimizer_bridge
        
#         self.quantization_wrappers = []
#         self.quantizable_modules = []
#         self.quantizable_layers = quantizable_layers
#         self._pre_process_container(model)
#         self._create_quantization_wrappers()
#         self.quantization_params = LearnedStepSizeQuantization.learned_parameters()

#     def load_state_dict(self, state_dict):
#         for name, qwrapper in self.quantization_wrappers:
#             qwrapper.load_state_dict(state_dict)

#     def freeze(self):
#         for n, p in self.model.named_parameters():
#             # TODO: hack, make it more robust
#             if not np.any([qp in n for qp in self.quantization_params]):
#                 p.requires_grad = False

#     @staticmethod
#     def has_children(module):
#         try:
#             next(module.children())
#             return True
#         except StopIteration:
#             return False
    
#     def _create_quantization_wrappers(self):
#         for qm in self.quantizable_modules:
#             # replace module by it's wrapper
#             fn = self.functor_map[type(qm.module)](qm.module) if type(qm.module) in self.functor_map else None
#             args = {"bits_out": self.bit_act, "bits_weight": self.bit_weights, "forward_functor": fn,
#                     "post_relu": self.post_relu, "optim_bridge": self.optimizer_bridge}
#             args.update(self.args)
#             if hasattr(qm, 'bn'):
#                 args['bn'] = qm.bn
#             module_wrapper = self.replacement_factory[type(qm.module)](qm.full_name, qm.module,
#                                                                     **args)
#             setattr(qm.container, qm.name, module_wrapper)
#             self.quantization_wrappers.append((qm.full_name, module_wrapper))

#     def _pre_process_container(self, container, prefix=''):
#         prev, prev_name = None, None
#         for name, module in container.named_children():
#             # if is_bn(module) and is_absorbing(prev) and prev_name in self.quantizable_layers:
#             #     # Pass BN module to prev module quantization wrapper for BN folding/unfolding
#             #     self.quantizable_modules[-1].bn = module

#             full_name = prefix + name
#             if full_name in self.quantizable_layers:
#                 self.quantizable_modules.append(
#                     type('', (object,), {'name': name, 'full_name': full_name, 'module': module, 'container': container})()
#                 )

#             if self.has_children(module):
#                 # For container we call recursively
#                 self._pre_process_container(module, full_name + '.')

#             prev = module
#             prev_name = full_name

#     def get_qwrappers(self):
#         return [qwrapper for (name, qwrapper) in self.quantization_wrappers if qwrapper.__enabled__()]

#     def set_clipping(self, clipping, device):  # TODO: handle device internally somehow
#         qwrappers = self.get_qwrappers()
#         for i, qwrapper in enumerate(qwrappers):
#             qwrapper.set_quantization(FixedClipValueQuantization,
#                                       {'clip_value': clipping[i], 'device': device})

#     def get_clipping(self):
#         clipping = []
#         qwrappers = self.get_qwrappers()
#         for i, qwrapper in enumerate(qwrappers):
#             q = qwrapper.get_quantization()
#             clip_value = getattr(q, 'alpha')
#             clipping.append(clip_value.item())

#         return qwrappers[0].get_quantization().alpha.new_tensor(clipping)