import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# LSQ 基线模型所需模块 (用于消融实验对比)
# ==========================================

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class LSQQuantizer(nn.Module):
    def __init__(self, bit_width, is_activation=False):
        super().__init__()
        self.bit_width = bit_width
        self.is_activation = is_activation

        # 定义量化的上下界
        if self.is_activation:
            self.Qn = 0
            self.Qp = 2 ** self.bit_width - 1
        else:
            self.Qn = -2 ** (self.bit_width - 1)
            self.Qp = 2 ** (self.bit_width - 1) - 1

        # LSQ 的核心：可学习的缩放步长 s
        self.s = nn.Parameter(torch.ones(1))
        self.init_state = False  # 标记是否已经初始化

    def forward(self, x):
        # LSQ 初始化逻辑：利用第一批数据统计绝对值均值来初始化 s
        if not self.init_state and self.training:
            init_val = 2 * x.detach().abs().mean() / math.sqrt(self.Qp)
            self.s.data.copy_(init_val.clamp(min=1e-4))
            self.init_state = True

        s_val = torch.abs(self.s) + 1e-8
        x_scaled = x / s_val
        x_rounded = RoundSTE.apply(x_scaled)
        x_clamped = torch.clamp(x_rounded, self.Qn, self.Qp)
        x_dequant = x_clamped * s_val

        return x_dequant


# 继承 nn.Linear，完美兼容你的 Teacher 权重加载逻辑
class LSQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, w_bit=4, a_bit=2):
        super().__init__(in_features, out_features, bias)
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.weight_quantizer = LSQQuantizer(bit_width=w_bit, is_activation=False)
        self.act_quantizer = LSQQuantizer(bit_width=a_bit, is_activation=True)

    def forward(self, input):
        # 分别对输入激活值和权重进行 LSQ 可学习量化
        q_x = self.act_quantizer(input)
        q_w = self.weight_quantizer(self.weight)
        return F.linear(q_x, q_w, self.bias)

# 1. 核心算法：STE 直通估计器 (支持分位数采样逻辑)

class STEQuantizer(torch.autograd.Function):
    """
    直通估计器 (Straight-Through Estimator)
    支持基于分位数的时序感知校正 (TQC) 的对称量化
    """
    @staticmethod
    def forward(ctx, input, bits, use_correction=True):
        # 将位宽信息存入 context 备用
        ctx.bits = bits

        # --- 核心逻辑：确定量化边界 v_max ---
        if use_correction:
            # 创新点 3：分位数校正 (TQC)
            # 针对显存优化的采样逻辑，防止对超大张量全量排序
            if input.numel() > 1000000:
                flat_input = input.detach().abs().view(-1)
                # 随机采样 100 万个点来估计分位数
                indices = torch.randint(0, flat_input.size(0), (1000000,), device=input.device)
                v_max = torch.quantile(flat_input[indices].float(), 0.999)
            else:
                v_max = torch.quantile(input.detach().abs().float().flatten(), 0.999)
        else:
            # 传统方式：基于最大值的量化 (常用于分布较稳定的权重)
            v_max = input.detach().abs().max()

        # 引入 epsilon 防止除零错误
        v_max = v_max + 1e-8

        # --- 对称量化计算 (Symmetrical) ---
        # 计算量化阶数（例如 2-bit 下，q_max = 1，范围 [-1, 1]）
        q_max = (1 << (bits - 1)) - 1
        scale = q_max / v_max

        # 1. 量化与截断 (Scale & Clamp)
        output = torch.clamp(torch.round(input * scale), -q_max, q_max)

        # 2. 反量化还原回浮点域 (De-quantization)
        output = output / (scale + 1e-8)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        STE 核心机制：
        在反向传播时跳过 round 函数的不可导部分，将梯度直接穿透传递
        """
        # 返回与 forward 输入参数对应的梯度 (grad_output 对应 input，后续 None 对应 bits 和 use_correction)
        return grad_output, None, None

# 2. 异构位宽线性层：实现 W4A2 的关键
class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bits=2, weight_bits=4, bias=True, enable_quant=True):
        super().__init__(in_features, out_features, bias)
        self.bits = bits
        self.weight_bits = weight_bits
        self.use_correction = True
        self.enable_quant = enable_quant

    def forward(self, input):
        # teacher / FP32 path
        if (not self.enable_quant) or (self.bits >= 32 and self.weight_bits >= 32):
            return F.linear(input, self.weight, self.bias)

        # quantized student path
        q_weight = STEQuantizer.apply(self.weight, self.weight_bits, False)
        return F.linear(input, q_weight, self.bias)


# 3. 激活值专用量化模块 (用于 EncoderLayer 中的 temporal_quantizer)
class TemporalAwareQuantizer(nn.Module):
    def __init__(self, bits=2):
        super().__init__()
        self.bits = bits
        self.use_correction = True  # 默认开启你的创新点逻辑

    def forward(self, x):
        if self.bits >= 32:
            return x
        return STEQuantizer.apply(x, self.bits, self.use_correction)


# 为了保持与 iTransformer 原有代码的兼容性
class QuantizerModule(TemporalAwareQuantizer):
    def __init__(self, bits=4):
        super().__init__(bits=bits)