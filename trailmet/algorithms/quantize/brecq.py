import torch
import torch.nn as nn
import torch.distributed as dist
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantization, BaseQuantModel#, StraightThrough
# from trailmet.models.resnet import BasicBlock, Bottleneck
# from trailmet.models.mobilenet import InvertedResidual
from trailmet.algorithms.quantize.qmodel import QuantModule, BaseQuantBlock
# from trailmet.algorithms.quantize.qmodel import QuantBasicBlock, QuantBottleneck, QuantInvertedResidual
from trailmet.algorithms.quantize.reconstruct import layer_reconstruction, block_reconstruction

# supported = {
#     BasicBlock: QuantBasicBlock,
#     Bottleneck: QuantBottleneck,
#     InvertedResidual: QuantInvertedResidual,
# }

class BRECQ(BaseQuantization):
    """
    Class for post-training quantization using block reconstruction method 
    based on - BRECQ: PUSHING THE LIMIT OF POST-TRAINING QUANTIZATION 
    BY BLOCK RECONSTRUCTION [https://arxiv.org/abs/2102.05426]

    :param W_BITS: bitwidth for weight quantization
    :param A_BITS: bitwidth for activation quantization
    :param CHANNEL_WISE: apply channel_wise quantization for weights
    :param ACT_QUANT: apply activation quantization
    :param SET_8BIT_HEAD_STEM: Set the first and the last layer to 8-bit
    :param NUM_SAMPLES: size of calibration dataset
    :param WEIGHT: weight of rounding cost vs the reconstruction loss
    :param ITERS_W: number of iteration for AdaRound
    :param ITERS_A: number of iteration for LSQ
    :param LR: learning rate for LSQ
    """
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(BRECQ, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['test']
        self.kwargs = kwargs
        self.w_bits = self.kwargs.get('W_BITS', 8)
        self.a_bits = self.kwargs.get('A_BITS', 8)
        self.channel_wise = self.kwargs.get('CHANNEL_WISE', True)
        self.act_quant = self.kwargs.get('ACT_QUANT', True)
        self.set_8bit_head_stem = self.kwargs.get('SET_8BIT_HEAD_STEM', False)
        self.w_budget = self.kwargs.get('W_BUDGET', None)
        self.use_bits = self.kwargs.get('USE_BITS', [2,4,8])
        self.arch = self.kwargs.get('ARCH', '')
        self.save_path = self.kwargs.get('SAVE_PATH', './runs/')
        self.num_samples = self.kwargs.get('NUM_SAMPLES', 1024)

        self.iters_w = self.kwargs.get('ITERS_W', 10000)
        self.iters_a = self.kwargs.get('ITERS_A', 10000)
        self.optimizer = self.kwargs.get('OPTIMIZER', 'adam')
        self.weight = self.kwargs.get('WEIGHT', 0.01)
        self.lr = self.kwargs.get('LR', 4e-4)

        self.gpu_id = self.kwargs.get('GPU_ID', 0)
        self.calib_bs = self.kwargs.get('CALIB_BS', 64)
        self.seed = self.kwargs.get('SEED', 42)
        self.p = 2.4         # Lp norm minimization for LSQ
        self.b_start = 20    # temperature at the beginning of calibration
        self.b_end = 2       # temperature at the end of calibration
        self.test_before_calibration = True
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        torch.cuda.set_device(self.gpu_id)
        seed_everything(self.seed)
        print('==> Using seed :',self.seed)


    def compress_model(self):
        """
        method to build quantization parameters and finetune weights and/or activations
        """
        wq_params = {
            'n_bits': self.w_bits, 
            'channel_wise': self.channel_wise, 
            'scale_method': 'mse'
        }
        aq_params = {
            'n_bits': self.a_bits, 
            'channel_wise': False, 
            'scale_method': 'mse', 
            'leaf_param': self.act_quant
        }
        self.model = self.model.to(self.device)
        self.model.eval()
        self.qnn = QuantModel(model=self.model, weight_quant_params=wq_params, act_quant_params=aq_params)
        self.qnn = self.qnn.to(self.device)
        self.qnn.eval()

        w_compr = self.w_bits/32 if self.w_budget is None else self.w_budget
        if self.w_budget is not None:
            w_bits, qm_size, max_size = self.sensitivity_analysis(
                self.qnn, self.test_loader, self.use_bits, self.w_budget, 
                self.save_path, '{}_{}_{}'.format(self.arch, w_compr, self.a_bits))
            print('==> Found optimal config for approx model size: {:.2f} MB \
                  (orig {:.2f} MB)'.format(qm_size, max_size/self.w_budget))
            self.qnn.set_layer_precision(w_bits, self.a_bits)
            self.qnn.reset_scale_method('mse', True)
        
        if self.set_8bit_head_stem:
            print('==> Setting the first and the last layer to 8-bit')
            self.qnn.set_head_stem_precision(8)

        self.cali_data = self.get_calib_samples(self.train_loader, self.num_samples)
        self.qnn.set_quant_state(True, False)
        print('==> Initializing weight quantization parameters')
        _ = self.qnn(self.cali_data[:self.calib_bs].to(self.device))
        if self.test_before_calibration:
            print('Quantized accuracy before brecq: {}'.format(self.test(self.qnn, self.test_loader, device=self.device)))
        
        # Start quantized weight calibration
        kwargs = dict(
            cali_data=self.cali_data, 
            iters=self.iters_w, 
            weight=self.weight, 
            asym=True,
            b_range=(self.b_start, self.b_end), 
            warmup=0.2, 
            act_quant=False, 
            opt_mode='mse', 
            optim=self.optimizer
        )
        print('==> Starting quantized-weight rounding parameter (alpha) calibration')
        self.reconstruct_model(self.qnn, **kwargs)
        self.qnn.set_quant_state(weight_quant=True, act_quant=False)
        print('Weight quantization accuracy: {}'.format(self.test(self.qnn, self.test_loader, device=self.device)))

        if self.act_quant:
            # Initialize activation quantization parameters
            self.qnn.set_quant_state(True, True)
            with torch.no_grad():
                _ = self.qnn(self.cali_data[:self.calib_bs].to(self.device))
            self.qnn.disable_network_output_quantization()
            
            # Start activation rounding calibration
            kwargs = dict(
                cali_data=self.cali_data, 
                iters=self.iters_a, 
                act_quant=True, 
                opt_mode='mse', 
                lr=self.lr, 
                p=self.p, 
                optim=self.optimizer
            )
            print('==> Starting quantized-activation scaling parameter (delta) calibration')
            self.reconstruct_model(self.qnn, **kwargs)
            self.qnn.set_quant_state(weight_quant=True, act_quant=True)
            print('Full quantization (W{}A{}) accuracy: {}'.format(w_compr, self.a_bits, 
                self.test(self.qnn, self.test_loader, device=self.device))) 
        return self.qnn


    def reconstruct_model(self, model: nn.Module, **kwargs):
        """
        Method for model parameters reconstruction. Takes in quantized model
        and optimizes weights by applying layer-wise reconstruction for first 
        and last layer, and block reconstruction otherwise.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(self.qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    block_reconstruction(self.qnn, module, **kwargs)
            else:
                self.reconstruct_model(module, **kwargs)


class QuantModel(BaseQuantModel):
    def __init__(self, model: nn.Module, weight_quant_params: dict, act_quant_params: dict):
        super(QuantModel, self).__init__(model, weight_quant_params, act_quant_params, fold_bn=True)

    def reset_scale_method(self, scale_method = 'mse', act_quant_reset = False):
        for module in self.quant_modules:
            module.weight_quantizer.scale_method = scale_method
            module.weight_quantizer.inited = False
            if act_quant_reset:
                module.act_quantizer.scale_method = scale_method
                module.act_quantizer.inited = False

    def quantize_model_till(self, layer, act_quant: bool = False):
        """
        :param layer: layer upto which model is to be quantized.
        :param act_quant: set True for activation quantization
        """
        self.set_quant_state(False, False)
        for name, module in self.model.named_modules():
            if isinstance(module, (QuantModule, BaseQuantBlock)):
                module.set_quant_state(True, act_quant)
            if module == layer:
                break 
    
    def set_head_stem_precision(self, bitwidth):
        """
        Set the precision (bitwidth) for weights and activations for the first and last 
        layers of the model. Also ignore reconstruction for the first layer.
        """
        assert len(self.quant_modules) >= 2, 'Model has less than 2 quantization modules'
        self.quant_modules[0].weight_quantizer.bitwidth_refactor(bitwidth)
        self.quant_modules[0].act_quantizer.bitwidth_refactor(bitwidth)
        self.quant_modules[-1].weight_quantizer.bitwidth_refactor(bitwidth)
        self.quant_modules[-2].act_quantizer.bitwidth_refactor(bitwidth)
        self.quant_modules[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        """
        Disable Network Output Quantization
        """
        self.quant_modules[-1].disable_act_quant = True

    def synchorize_activation_statistics(self):
        """
        Synchronize the statistics of the activation quantizers across all distributed workers.
        """
        for m in self.modules():
            if isinstance(m, QuantModule):
                if m.act_quantizer.delta is not None:
                    m.act_quantizer.delta.data /= dist.get_world_size()
                    dist.all_reduce(m.act_quantizer.delta.data)
    

# class QuantModel(nn.Module):
#     """
#     Recursively replace the normal conv2d and Linear layer to QuantModule, to enable 
#     calculating activation statistics and storing scaling factors.

#     :param module: nn.Module with nn.Conv2d or nn.Linear in its children
#     :param weight_quant_params: quantization parameters like n_bits for weight quantizer
#     :param act_quant_params: quantization parameters like n_bits for activation quantizer
#     """
#     def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
#         super().__init__()
#         self.model = model
#         bn = FoldBN()
#         bn.search_fold_and_remove_bn(self.model)
#         self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
#         self.quant_modules = [m for m in self.model.modules() if isinstance(m, QuantModule)]

#     def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
#         prev_quantmodule = None
#         for name, child_module in module.named_children():
#             if type(child_module) in supported:
#                 setattr(module, name, supported[type(child_module)](child_module, weight_quant_params, act_quant_params))

#             elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
#                 setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
#                 prev_quantmodule = getattr(module, name)

#             elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
#                 if prev_quantmodule is not None:
#                     prev_quantmodule.activation_function = child_module
#                     setattr(module, name, StraightThrough())
#                 else:
#                     continue

#             elif isinstance(child_module, StraightThrough):
#                 continue

#             else:
#                 self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)
    
#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         """
#         :param weight_quant: set True for weight quantization
#         :param act_quant: set True for activation quantization
#         """
#         for m in self.model.modules():
#             if isinstance(m, (QuantModule, BaseQuantBlock)):
#                 m.set_quant_state(weight_quant, act_quant)

#     def quantize_model_till(self, layer, act_quant: bool = False):
#         """
#         :param layer: block/layer upto which model is to be quantized.
#         :param act_quant: set True for activation quantization
#         """
#         self.set_quant_state(False, False)
#         for name, module in self.model.named_modules():
#             if isinstance(module, (QuantModule, BaseQuantBlock)):
#                 module.set_quant_state(True, act_quant)
#             if module == layer:
#                 break

#     def forward(self, input):
#         return self.model(input)

#     def set_first_last_layer_to_8bit(self):
#         """
#         Set the precision (bitwidth) used for quantizing weights and activations to 8-bit
#         for the first and last layers of the model. Also ignore reconstruction for the first layer.
#         """
#         assert len(self.quant_modules) >= 2, 'Model has less than 2 quantization modules'
#         self.quant_modules[0].weight_quantizer.bitwidth_refactor(8)
#         self.quant_modules[0].act_quantizer.bitwidth_refactor(8)
#         self.quant_modules[-1].weight_quantizer.bitwidth_refactor(8)
#         self.quant_modules[-2].act_quantizer.bitwidth_refactor(8)
#         self.quant_modules[0].ignore_reconstruction = True

#     def disable_network_output_quantization(self):
#         self.quant_modules[-1].disable_act_quant = True

#     def set_layer_precision(self, weight_bit=8, act_bit=8, start=0, end=None):
#         """
#         Set the precision (bitwidth) used for quantizing weights and activations
#         for a range of layers in the model.

#         :param weight_bit: number of bits to use for quantizing weights
#         :param act_bit: number of bits to use for quantizing activations
#         :param start: index of the first layer to set the precision for (default: 0)
#         :param end: index of the last layer to set the precision for (default: None, i.e., the last layer)
#         """
#         assert start>=0 and end>=0, 'layer index cannot be negative'
#         assert start<len(self.quant_modules) and end<len(self.quant_modules), 'layer index out of range'
        
#         for module in self.quant_modules[start: end+1]:
#             module.weight_quantizer.bitwidth_refactor(weight_bit)
#             if module is not self.quant_modules[-1]:
#                 module.act_quantizer.bitwidth_refactor(act_bit)

#     def synchorize_activation_statistics(self):
#         """
#         Synchronize the statistics of the activation quantizers across all distributed workers.
#         """
#         for m in self.modules():
#             if isinstance(m, QuantModule):
#                 if m.act_quantizer.delta is not None:
#                     m.act_quantizer.delta.data /= dist.get_world_size()
#                     dist.all_reduce(m.act_quantizer.delta.data) 