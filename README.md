Forecast-Preserving Low-Bit Quantization for Transformer-Based Time-Series Forecasting

This repository contains the implementation of a low-bit quantization-aware training framework for multivariate time-series forecasting based on the inverted Transformer (iTransformer). The framework combines percentile-guided activation range estimation with forecast-level knowledge distillation to improve forecasting robustness under aggressive low-bit weight-and-activation quantization.

Main Components

Percentile-Guided Dynamic Range Quantization: estimates activation clipping boundaries using high-percentile statistics to reduce the influence of rare temporal outliers.

Forecast-Preserving Distillation: aligns the forecasting output of the quantized student with that of a pretrained full-precision teacher.

Quantization-Aware Training: applies fake quantization to the weights and activations of the student model during training.

Efficient Inference: only the quantized student is retained during inference; the full-precision teacher and distillation loss are not required.

Repository Structure

.
├── run.py
├── models/
├── layers/
├── experiments/
├── utils/
├── scripts/
├── dataset/
├── checkpoints/
└── README.md

The exact directory names may differ slightly depending on the released version of the repository.

Environment Setup

Create and activate a Python environment:

conda create -n lowbit_forecasting python=3.10 -y
conda activate lowbit_forecasting

Install the required packages:

pip install -r requirements.txt

Please update requirements.txt with the package versions used in the experiments before releasing the repository.

Dataset Preparation

Download the public forecasting datasets and place them under the dataset directory. For example, the Electricity dataset should be organized as follows:

dataset/
└── electricity/
    └── electricity.csv

The dataset path is specified using:

--root_path ./dataset/electricity/
--data_path electricity.csv

The released repository should provide the original download sources or preparation instructions for all datasets used in the paper.

Full-Precision Teacher Checkpoint

Forecast-Preserving Distillation requires a pretrained full-precision iTransformer model as the teacher during quantization-aware training.

Before running the quantized training command, either:

place the pretrained teacher checkpoint at a predefined relative path, such as

./checkpoints/fp32/ECL_96/checkpoint.pth

or

modify --teacher_path so that it points to the actual checkpoint location on your machine.

For example:

--teacher_path ./checkpoints/fp32/ECL_96/checkpoint.pth

Do not use the original absolute path from another server, such as /home/username/..., because it will not be valid on other machines.

Example: ECL, Prediction Length 96, W4A4

The following command runs quantization-aware training on the Electricity dataset with 4-bit weights and 4-bit activations:

python run.py \
  --is_training 1 \
  --model iTransformer \
  --model_id ECL_96 \
  --data custom \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 8 \
  --e_layers 2 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 2 \
  --attn_bits 4 \
  --ffn_bits 4 \
  --itr 1 \
  --teacher_path ./checkpoints/fp32/ECL_96/checkpoint.pth \
  --checkpoints ./checkpoints/ECL_W4A4_96

Here:

--attn_bits 4 sets the bit width used in the attention module.

--ffn_bits 4 sets the bit width used in the feed-forward network.

--teacher_path specifies the pretrained full-precision teacher checkpoint.

--checkpoints specifies where the quantized student checkpoint will be saved.

--itr 1 specifies one experimental run.

Other Bit-Width Settings

To run other quantization settings, change both --attn_bits and --ffn_bits.

W8A8

--attn_bits 8 \
--ffn_bits 8

W4A4

--attn_bits 4 \
--ffn_bits 4

W2A2

--attn_bits 2 \
--ffn_bits 2

In the paper, W and A denote the bit widths of weights and activations, respectively.

Changing the Prediction Horizon

Set the forecasting horizon using --pred_len. Typical settings include:

--pred_len 96
--pred_len 192
--pred_len 336
--pred_len 720

The corresponding --model_id, teacher checkpoint path, and output directory should also be changed to avoid overwriting previous experiments.

For example:

--model_id ECL_192 \
--pred_len 192 \
--teacher_path ./checkpoints/fp32/ECL_192/checkpoint.pth \
--checkpoints ./checkpoints/ECL_W4A4_192

Training and Inference

During training:

the student model uses fake-quantized weights and activations;

the teacher model remains in full precision;

the teacher provides forecasting-output supervision;

gradients are updated only for the student model.

During inference:

the teacher model is not loaded;

the distillation loss is not computed;

only the quantized student model is retained.

Therefore, Forecast-Preserving Distillation introduces no additional teacher-model overhead during deployment.

Reproducibility

To reproduce the reported results, keep the following settings consistent across all compared quantization methods:

dataset split;

input sequence length;

prediction horizon;

model architecture;

pretrained full-precision checkpoint;

batch size;

optimizer and learning rate;

training epochs;

random seed;

quantized module scope;

weight and activation bit widths.

The final released version should include:

requirements.txt
scripts/
configs/

and should provide one execution script or configuration file for every main table in the paper.

Expected Outputs

Model checkpoints are saved under the directory specified by:

--checkpoints

Training and evaluation logs should report at least:

mean squared error;

mean absolute error;

quantization bit widths;

dataset name;

prediction horizon;

checkpoint path.

Citation

Please cite the paper after publication:

@article{your_citation_key,
  title   = {Forecast-Preserving Low-Bit Quantization for Transformer-Based Time-Series Forecasting},
  author  = {Author names},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {Year}
}

Replace the placeholder bibliographic information with the final publication details.

Code Availability

The source code, configuration files, and execution scripts used to reproduce the experiments are publicly available in this repository.

Acknowledgment

This implementation is developed based on the official iTransformer forecasting codebase. Please also cite the original iTransformer paper and repository when using this code.