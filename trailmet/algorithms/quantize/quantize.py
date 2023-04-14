import os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from plotly import graph_objects
from trailmet.models.resnet import BasicBlock, Bottleneck
from trailmet.models.mobilenet import InvertedResidual
from trailmet.algorithms.quantize.qmodel import QuantModule, BaseQuantBlock
from trailmet.algorithms.quantize.qmodel import QuantBasicBlock, QuantBottleneck, QuantInvertedResidual
from trailmet.algorithms.algorithms import BaseAlgorithm

class Node:
    def __init__(self, cost=0, profit=0, bit=None, parent=None, left=None, middle=None, right=None, position='middle'):
        self.parent = parent
        self.left = left
        self.middle = middle
        self.right = right
        self.position = position
        self.cost = cost
        self.profit = profit
        self.bit = bit

    def __str__(self):
        return 'cost: {:.2f} profit: {:.2f}'.format(self.cost, self.profit)
    
    def __repr__(self):
        return self.__str__()


supported = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    InvertedResidual: QuantInvertedResidual,
}
class BaseQuantModel(nn.Module):
    """base model wrapping class for quantization algorithms"""
    def __init__(self, model: nn.Module, weight_quant_params: dict = {},
            act_quant_params: dict = {}, fold_bn = True):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        if fold_bn:
            self.model.eval()
            self.search_fold_remove_bn(self.model)
        self.quant_module_refactor(self.model)
        self.quant_modules = [m for m in self.model.modules() if isinstance(m, QuantModule)]


    def search_fold_remove_bn(self, module: nn.Module):
        """
        Recursively search for BatchNorm layers, fold them into the previous 
        Conv2d or Linear layers and set them to a StraightThrough layer.
        """
        prev_module = None
        for name, child_module in module.named_children():
            if self._is_bn(child_module) and self._is_absorbing(prev_module):
                self._fold_bn_into_conv(prev_module, child_module)
                setattr(module, name, StraightThrough())
            elif self._is_absorbing(child_module):
                prev_module = child_module
            else:
                prev_module = self.search_fold_remove_bn(child_module)
        return prev_module

    def quant_module_refactor(self, module: nn.Module):
        """
        Recursively replace Conv2d and Linear layers with QuantModule and other 
        supported network blocks to their respective wrappers, to enable weight 
        and activations quantization.
        """
        prev_quant_module: QuantModule = None
        for name, child_module in module.named_children():
            if type(child_module) in supported:
                setattr(module, name, supported[type(child_module)](
                    child_module, self.weight_quant_params, self.act_quant_params
                ))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(
                    child_module, self.weight_quant_params, self.act_quant_params
                ))
                prev_quant_module = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quant_module is not None:
                    prev_quant_module.activation_function = child_module
                else:
                    continue
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module)
            

    def set_quant_state(self, weight_quant: bool = True, act_quant: bool = True):
        """
        :param weight_quant: set True to enable weight quantization
        :param act_quant: set True to enable activation quantization
        """
        for module in self.model.modules():
            if isinstance(module, (QuantModule, BaseQuantBlock)):
                module.set_quant_state(weight_quant, act_quant)

    def set_layer_precision(self, weight_bits: list, act_bit: int):
        """
        :param weight_bits: list of bitwidths for layer weights
        :param act_bit: bitwidth for activations
        """
        assert len(weight_bits)==len(self.quant_modules)
        for idx, module in  enumerate(self.quant_modules):
            module.weight_quantizer.bitwidth_refactor(weight_bits[idx])
            if module is not self.quant_modules[-1]:
                module.act_quantizer.bitwidth_refactor(act_bit)

    def forward(self, input):
        return self.model(input)
    
    def _is_bn(self, module):
        return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    
    def _is_absorbing(self, module):
        return isinstance(module, (nn.Conv2d, nn.Linear))
    
    def _fold_bn_into_conv(self, conv_module, bn_module):
        w, b = self._get_folded_params(conv_module, bn_module)
        if conv_module.bias is None:
            conv_module.bias = nn.Parameter(b)
        else:
            conv_module.bias.data = b
        conv_module.weight.data = w
        bn_module.running_mean = bn_module.bias.data
        bn_module.running_var = bn_module.weight.data ** 2

    def _get_folded_params(self, conv_module, bn_module):
        w = conv_module.weight.data
        y_mean = bn_module.running_mean
        y_var = bn_module.running_var
        safe_std = torch.sqrt(y_var + bn_module.eps)
        w_view = (conv_module.out_channels, 1, 1, 1)
        if bn_module.affine:
            weight = w * (bn_module.weight / safe_std).view(w_view)
            beta = bn_module.bias - bn_module.weight * y_mean / safe_std
            if conv_module.bias is not None:
                bias = bn_module.weight * conv_module.bias / safe_std + beta
            else:
                bias = beta
        else:
            weight = w / safe_std.view(w_view)
            beta = -y_mean / safe_std
            if conv_module.bias is not None:
                bias = conv_module.bias / safe_std + beta
            else:
                bias = beta
        return weight, bias




class BaseQuantization(BaseAlgorithm):
    """base class for quantization algorithms"""
    def __init__(self, **kwargs):
        super(BaseQuantization, self).__init__(**kwargs)
        pass

    def quantize(self, model, dataloaders, method, **kwargs):
        pass

    def get_calib_samples(self, train_loader, num_samples):
        """
        Get calibration-set samples for finetuning weights and clipping parameters
        """
        calib_data = []
        for batch in train_loader:
            calib_data.append(batch[0])
            if len(calib_data)*batch[0].size(0) >= num_samples:
                break
        return torch.cat(calib_data, dim=0)[:num_samples]

    # def absorb_bn(self, module, bn_module):
    #     w = module.weight.data
    #     if module.bias is None:
    #         zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
    #         module.bias = nn.Parameter(zeros)
    #     b = module.bias.data
    #     invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
    #     w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
    #     b.add_(-bn_module.running_mean).mul_(invstd)

    #     if bn_module.affine:
    #         w.mul_(bn_module.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
    #         b.mul_(bn_module.weight.data).add_(bn_module.bias.data)

    #     bn_module.register_buffer('running_mean', torch.zeros(module.out_channels).cuda())
    #     bn_module.register_buffer('running_var', torch.ones(module.out_channels).cuda())
    #     bn_module.register_parameter('weight', None)
    #     bn_module.register_parameter('bias', None)
    #     bn_module.affine = False

    # def is_bn(self, m):
    #     return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)

    # def is_absorbing(self, m):
    #     return (isinstance(m, nn.Conv2d) and m.groups == 1) or isinstance(m, nn.Linear)

    # def search_absorbe_bn(self, model):
    #     prev = None
    #     for m in model.children():
    #         if self.is_bn(m) and self.is_absorbing(prev):
    #             m.absorbed = True
    #             self.absorb_bn(prev, m)
    #         self.search_absorbe_bn(m)
    #         prev = m

    def sensitivity_analysis(self, qmodel: BaseQuantModel, dataloader, test_bits, budget, exp_name):
        qmodel.set_quant_state(False, False)
        inputs = None
        fp_outputs = None
        with torch.no_grad():
            for batch_idx, (inputs, outputs) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                fp_outputs = qmodel(inputs)
                fp_outputs = F.softmax(fp_outputs, dim=1)
                break
        sensitivities = [[0 for i in range(len(qmodel.quant_modules))] for j in range(len(test_bits))]
        for i, layer in enumerate(qmodel.quant_modules):
            for j, bit in enumerate(test_bits):
                layer.set_quant_state(True, True)
                layer.weight_quantizer.bitwidth_refactor(bit)
                layer.weight_quantizer.inited = False
                layer.weight_quantizer.scale_method = 'max'
                with torch.no_grad():
                    tmp_outputs = qmodel(inputs)
                    tmp_outputs = F.softmax(tmp_outputs, dim=1)
                    kld = symmetric_kl_div(tmp_outputs, fp_outputs)
                sensitivities[j][i] = kld.item()
                layer.set_quant_state(False, False)
        plot_layer_sensitivity(sensitivities, test_bits, exp_name)

        weight_numels = [qmodule.weight.numel() for qmodule in qmodel.quant_modules]
        node_list = self.dp_most_profit_over_cost(sensitivities, len(qmodel.quant_modules), weight_numels)
        constraint = sum(weight_numels)*32*budget / (8*24*24)
        good_nodes = [node for node in node_list if node.cost <= constraint]
        bits = []
        node = good_nodes[-1]
        while(node is not None):
            bits.append(node.bit)
            node = node.parent
        bits.reverse()
        bits = bits[1:]
        assert len(bits)==len(qmodel.quant_modules)
        plot_layer_precisions(bits, exp_name)
        qmodel_size = 0
        for i, layer in enumerate(qmodel.quant_modules):
            qmodel_size += layer.weight.numel()*bits[i]/(8*1024*1024)
        return bits, qmodel_size, constraint

    def dp_most_profit_over_cost(self, sensitivities, num_layers, weight_numels, bits, constraint=100):
        cost = bits
        profits = []
        for line in sensitivities:
            profits.append([-i for i in line])
        root = Node(cost=0, profit=0, parent=None)
        current_list = [root]
        for layer_id in range(num_layers):
            next_list = []
            for n in current_list:
                n.left = Node(n.cost + cost[0]*weight_numels[layer_id]/(8*1024*1024), 
                                n.profit + profits[0][layer_id],
                                bit = bits[0], parent=n, position='left')
                n.middle = Node(n.cost + cost[1]*weight_numels[layer_id]/(8*1024*1024), 
                                n.profit + profits[1][layer_id],
                                bit = bits[1], parent=n, position='middle')
                n.right = Node(n.cost + cost[2]*weight_numels[layer_id]/(8*1024*1024), 
                                n.profit + profits[2][layer_id],
                                bit = bits[2], parent=n, position='right')
                next_list.extend([n.left, n.middle, n.right])
            next_list.sort(key=lambda x: x.cost, reverse=False)
            pruned_list = []
            for node in next_list:
                if (len(pruned_list)==0 or pruned_list[-1].profit < node.profit) and node.cost <= constraint:
                    pruned_list.append(node)
                else:
                    node.parent.__dict__[node.position] = None
            current_list = pruned_list
        return current_list
    
    # def round_ste(x: torch.Tensor):
    #     """
    #     Implement Straight-Through Estimator for rounding operation.
    #     """
    #     return (x.round() - x).detach() + x

def plot_layer_sensitivity(senitivities, test_bits, exp_name):
    data = [graph_objects.Scatter(
        y = senitivities[i],
        mode = 'lines + markers',
        name = f'{bit}bit'
    ) for i, bit in enumerate(test_bits)]
    layout = graph_objects.Layout(
        title = '{} sensitivity analysis'.format(exp_name),
        xaxis = dict(title='layer'),
        yaxis = dict(title='sensitivity of quantization', type='log')
    )
    fig = graph_objects.Figure(data, layout)
    if not os.path.exists('./logs/plots'):
        os.mkdir('./logs/plots')
    fig.write_image('./logs/plots/{}_sensitivities.png'.format(exp_name))


def plot_layer_precisions(bits_, exp_name):
    data = [graph_objects.Scatter(
        y = bits_,
        mode = 'lines + markers'
    )]
    layout = graph_objects.Layout(
        title = exp_name,
        xaxis = dict(title='layer'),
        yaxis = dict(title='weight bitwidth')
    )
    fig = graph_objects.Figure(data, layout)
    if not os.path.exists('./logs/plots'):
        os.mkdir('./logs/plots')
    fig.write_image('./logs/plots/{}_bitwidths.png'.format(exp_name))


def kl_divergence(P, Q):
    return (P * (P/Q).log()).sum() / P.size(0)

def symmetric_kl_div(P, Q):
    return (kl_divergence(P, Q) + kl_divergence(Q, P)) / 2
    

class StraightThrough(nn.Module):
    """Identity Layer"""
    def __int__(self):
        super().__init__()
        pass

    def forward(self, input):
        return input


class RoundSTE(torch.autograd.Function):
    """Grad enabled rounding"""
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output 


class Conv2dFunctor:
    def __init__(self, conv2d):
        self.conv2d = conv2d
    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.conv2d(
            *input, weight, bias, 
            self.conv2d.stride, self.conv2d.padding,
            self.conv2d.dilation, self.conv2d.groups
        )
        return res

class LinearFunctor:
    def __init__(self, linear):
        self.linear = linear

    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.linear(*input, weight, bias)
        return res

# TODO : To migrate all BN-layer folding function calls to the ones defined inside BaseQuantization class 
class FoldBN():
    """used to fold batch norm to prev linear or conv layer which helps reduce comutational overhead during quantization"""
    def __init__(self):
        pass

    def _fold_bn(self, conv_module, bn_module):
        w = conv_module.weight.data
        y_mean = bn_module.running_mean
        y_var = bn_module.running_var
        safe_std = torch.sqrt(y_var + bn_module.eps)
        w_view = (conv_module.out_channels, 1, 1, 1)
        if bn_module.affine:
            weight = w * (bn_module.weight / safe_std).view(w_view)
            beta = bn_module.bias - bn_module.weight * y_mean / safe_std
            if conv_module.bias is not None:
                bias = bn_module.weight * conv_module.bias / safe_std + beta
            else:
                bias = beta
        else:
            weight = w / safe_std.view(w_view)
            beta = -y_mean / safe_std
            if conv_module.bias is not None:
                bias = conv_module.bias / safe_std + beta
            else:
                bias = beta
        return weight, bias


    def fold_bn_into_conv(self, conv_module, bn_module):
        w, b = self._fold_bn(conv_module, bn_module)
        if conv_module.bias is None:
            conv_module.bias = nn.Parameter(b)
        else:
            conv_module.bias.data = b
        conv_module.weight.data = w
        # set bn running stats
        bn_module.running_mean = bn_module.bias.data
        bn_module.running_var = bn_module.weight.data ** 2


    def is_bn(self, m):
        return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


    def is_absorbing(self, m):
        return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


    def search_fold_and_remove_bn(self, model: nn.Module):
        """
        method to recursively search for batch norm layers, absorb them into 
        the previous linear or conv layers, and set it to an identity layer 
        """
        model.eval()
        prev = None
        for n, m in model.named_children():
            if self.is_bn(m) and self.is_absorbing(prev):
                self.fold_bn_into_conv(prev, m)
                # set the bn module to straight through
                setattr(model, n, StraightThrough())
            elif self.is_absorbing(m):
                prev = m
            else:
                prev = self.search_fold_and_remove_bn(m)
        return prev
