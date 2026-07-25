from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import nullcontext

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        # Q-iTrans default setting:
        # FPD and PDRQ are always enabled.
        self.args.distil_quant = True
        self.args.use_correction = True

        print(">>> [Q-iTrans] Loading Teacher Model for Forecast-Preserving Distillation...")

        import copy
        teacher_args = copy.deepcopy(self.args)

        # Teacher is always full-precision.
        teacher_args.use_quant = False
        teacher_args.use_correction = False
        teacher_args.attn_bits = 32
        teacher_args.ffn_bits = 32
        teacher_args.distil = True

        # Align teacher architecture with student.
        teacher_args.e_layers = self.args.e_layers
        teacher_args.d_model = self.args.d_model
        teacher_args.d_ff = self.args.d_ff
        teacher_args.n_heads = self.args.n_heads

        self.teacher_model = self.model_dict[self.args.model].Model(teacher_args).float()

        teacher_path = self.args.teacher_path
        if os.path.exists(teacher_path):
            state_dict = torch.load(teacher_path, map_location=self.device)
            new_state_dict = {}

            for k, v in state_dict.items():
                # Convert Conv1d weights [out, in, 1] to Linear weights [out, in]
                # when loading old FP32 checkpoints.
                if ('.conv1.weight' in k or '.conv2.weight' in k) and v.dim() == 3:
                    new_state_dict[k] = v.squeeze(-1)
                else:
                    new_state_dict[k] = v

            self.teacher_model.load_state_dict(new_state_dict)
            print(f">>> Successfully loaded teacher weights from {teacher_path}")
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_path}")

        self.teacher_model.to(self.device)
        self.teacher_model.eval()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark, batch_y_mark = None, None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if outputs.numel() != target.numel():
                    outputs = outputs.reshape(1, self.args.pred_len, -1).repeat(
                        target.shape[0], 1, 1
                    )

                outputs = outputs.reshape(target.shape)
                loss = criterion(outputs.detach().cpu(), target.detach().cpu())
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Always enable percentile-guided correction for modules with use_correction.
        def enable_correction(m):
            if hasattr(m, 'use_correction'):
                m.use_correction = True

        self.model.apply(enable_correction)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count, train_loss = 0, []
            self.model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark, batch_y_mark = None, None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                amp_context = torch.cuda.amp.autocast() if self.args.use_amp else nullcontext()

                with amp_context:
                    res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    if isinstance(res, tuple):
                        student_outputs = res[0]
                        student_attns = res[1] if len(res) > 1 else None
                    else:
                        student_outputs = res
                        student_attns = None

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y_crop = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if student_outputs.numel() != batch_y_crop.numel():
                        if student_outputs.numel() == (self.args.pred_len * batch_y_crop.shape[-1]):
                            student_outputs = student_outputs.reshape(
                                1, self.args.pred_len, -1
                            ).repeat(batch_y_crop.shape[0], 1, 1)

                    student_outputs = student_outputs.reshape(batch_y_crop.shape)

                    loss_mse = criterion(student_outputs, batch_y_crop)

                    # FPD is always enabled.
                    with torch.no_grad():
                        t_res = self.teacher_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        if isinstance(t_res, tuple):
                            teacher_outputs = t_res[0]
                        else:
                            teacher_outputs = t_res

                        if teacher_outputs.numel() != batch_y_crop.numel():
                            if teacher_outputs.numel() == (self.args.pred_len * batch_y_crop.shape[-1]):
                                teacher_outputs = teacher_outputs.reshape(
                                    1, self.args.pred_len, -1
                                ).repeat(batch_y_crop.shape[0], 1, 1)

                        teacher_outputs = teacher_outputs.reshape(batch_y_crop.shape)

                    loss_distil_out = criterion(student_outputs, teacher_outputs)

                    # Final Q-iTrans objective.
                    loss = 0.8 * loss_mse + 0.2 * loss_distil_out

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()
                    ))

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(
                epoch + 1, train_loss, vali_loss
            ))

            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(path + '/' + 'checkpoint.pth'))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        print(f">>> Start testing: {len(test_loader)} batches...")

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark, batch_y_mark = None, None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if isinstance(res, tuple):
                    outputs = res[0]
                    student_attns = res[1] if len(res) > 1 else None
                else:
                    outputs = res
                    student_attns = None

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if outputs.numel() != batch_y.numel():
                    if outputs.numel() == (self.args.pred_len * batch_y.shape[-1]):
                        outputs = outputs.reshape(1, self.args.pred_len, -1).repeat(
                            batch_y.shape[0], 1, 1
                        )

                outputs = outputs.reshape(batch_y.shape)

                if i == 0 and student_attns is not None:
                    self._visualize_attention_comparison(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                        student_attns,
                        folder_path
                    )

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, self.args.pred_len, trues.shape[-1])
        trues = trues.reshape(-1, self.args.pred_len, trues.shape[-1])

        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{:.7f}, mae:{:.7f}'.format(mse, mae))

        result_path = './results/' + setting + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        np.save(result_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(result_path + 'pred.npy', preds)
        np.save(result_path + 'true.npy', trues)

        return

    def _visualize_attention_comparison(
        self,
        batch_x,
        batch_x_mark,
        dec_inp,
        batch_y_mark,
        student_attns,
        folder_path
    ):
        self.teacher_model.eval()

        with torch.no_grad():
            t_res = self.teacher_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if isinstance(t_res, tuple) and len(t_res) > 1:
                teacher_attns = t_res[1]
            else:
                teacher_attns = None

        if teacher_attns is not None and student_attns is not None:
            t_map = teacher_attns[0][0, 0].cpu().numpy()
            s_map = student_attns[0][0, 0].cpu().numpy()

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            sns.heatmap(t_map, cmap='viridis', vmin=0, vmax=0.05)
            plt.title("Teacher Attention (FP32)")

            plt.subplot(1, 2, 2)
            sns.heatmap(s_map, cmap='viridis', vmin=0, vmax=0.05)
            plt.title(f"Student Attention ({self.args.ffn_bits}-bit)")

            plt.savefig(os.path.join(folder_path, 'attn_comparison.png'))
            plt.close()