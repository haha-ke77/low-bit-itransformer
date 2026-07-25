import argparse
import torch
import random
import numpy as np
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial


# 定义一个专门设置随机种子的函数，确保全流程可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 针对老师要求的稳定性测试，必须开启确定性卷积算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f">>>> 已设置随机种子: {seed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iTransformer')

    # --- 基础配置 ---
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer', help='model name')

    # --- Data Loader ---
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')

    # --- 预测任务 ---
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # --- 模型参数 ---
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='output attention')

    # --- 优化与训练 ---
    parser.add_argument('--itr', type=int, default=1, help='实验运行次数，建议设置为 5 以响应老师建议')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')

    # --- GPU ---
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids')

    # --- iTransformer 特有参数 ---
    parser.add_argument('--exp_name', type=str, default='MTSF', help='MTSF, partial_train')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    # 在 run.py 的 parser 部分添加这一行
    parser.add_argument('--use_norm', type=int, default=1, help='use norm and denorm')
    # 在 run.py 中添加这个缺失的参数
    parser.add_argument('--embed', type=str, default='timeF',help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # --- 量化与蒸馏专项配置 ---
    parser.add_argument('--attn_bits', type=int, default=8, help='Attention bits')
    parser.add_argument('--ffn_bits', type=int, default=4, help='FFN bits')
    #parser.add_argument('--distil_quant', type=str, default='True', help='Use distillation')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    # 数据加载相关核心参数（必须补齐）
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    # 在 run.py 中补上自动混合精度开关
    parser.add_argument('--use_amp', type=str, default='True', help='use automatic mixed precision training')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    #parser.add_argument('--use_correction', type=str, default='True', help='Use TQC Innovation')
    parser.add_argument('--teacher_path', type=str, default='', help='Teacher model path')
    # 在 run.py 中补上模型保存路径参数
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--use_quant', type=str, default='True', help='whether to enable quantized forward')
    args = parser.parse_args()




    # 布尔逻辑转换
    args.use_quant = True if args.use_quant.lower() == 'true' else False
    #args.distil_quant = True if args.distil_quant.lower() == 'true' else False
    #args.use_correction = True if args.use_correction.lower() == 'true' else False
    args.use_amp = True if args.use_amp.lower() == 'true' else False
    args.distil_quant = True
    args.use_correction = True
    #if args.distil_quant:
        #args.output_attention = True

    # 准备 5 个不同的种子用于稳定性测试
    # 这样每次运行 ii 都会对应一个固定的种子
    random_seeds = [2022, 2023, 2024, 2025, 2026]

    if args.is_training:
        for ii in range(args.itr):
            # 1. 动态设置种子
            current_seed = random_seeds[ii] if ii < len(random_seeds) else 2023 + ii
            set_seed(current_seed)

            # 2. 设置实验标识 (加入 seed 标识以防结果文件被覆盖)
            setting = '{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_ab{}_fb{}_dq{}_uc{}_seed{}_{}'.format(
                args.model_id, args.model, args.data, args.features,
                args.seq_len, args.pred_len, args.d_model, args.n_heads, args.e_layers,
                args.attn_bits, args.ffn_bits, args.distil_quant, args.use_correction,
                current_seed, ii)

            # 3. 执行实验
            Exp = Exp_Long_Term_Forecast_Partial if args.exp_name == 'partial_train' else Exp_Long_Term_Forecast
            exp = Exp(args)

            print(f'>>>>>>> Start Loop {ii} | Seed {current_seed} | Setting: {setting} >>>>>>>')
            exp.train(setting)
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        # 测试模式直接运行单次
        set_seed(2023)
        setting = 'test_mode_setting'  # 简化的标识
        exp = Exp_Long_Term_Forecast(args)
        exp.test(setting, test=1)
        torch.cuda.empty_cache()