#!/bin/bash

# SimVP SCWDS 全流程脚本
# 包含: Train (SimVP) -> Test (SimVP) -> Infer (SimVP)
# Usage: bash run.scwds.simvp.sh [MODE]

# ================= 环境变量优化 =================
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# ================= 参数检查 =================
if [ $# -eq 0 ]; then
    echo "错误: 请指定操作模式"
    echo "用法: bash run.scwds.simvp.sh [MODE]"
    echo "支持的模式:"
    echo " train      - 训练 SimVP 基座模型"
    echo " test       - 测试 SimVP 基座模型"
    echo " infer      - 使用 SimVP 基座进行推理"
    exit 1
fi

MODE=$1

case $MODE in
    # ============================================================
    # 1. 训练 SimVP 基座 (Stage 1) - [保持原样]
    # ============================================================
    "train")
        echo "--------------------------------------------------------"
        echo "🚀 开始训练 Mamba 基座模型 (BF16 Mixed)..."
        echo "--------------------------------------------------------"
        
        python run/train_scwds_simvp.py \
            --data_path data/samples.jsonl \
            --save_dir ./output/simvp \
            --batch_size 1 \
            --accumulate_grad_batches 8 \
            --num_workers 8 \
            --in_shape 10 54 256 256 \
            --aft_seq_length 20 \
            --max_epochs 50 \
            --opt adamw \
            --lr 5e-6 \
            --sched cosine \
            --min_lr 1e-6 \
            --warmup_epoch 0 \
            --model_type mamba \
            --hid_S 128 \
            --hid_T 512 \
            --N_S 4 \
            --N_T 12 \
            --mlp_ratio 8.0 \
            --drop 0.0 \
            --drop_path 0.1 \
            --spatio_kernel_enc 5 \
            --spatio_kernel_dec 5 \
            --use_curriculum_learning false \
            --early_stop_patience 15 \
            --loss_weight_l1 0.1 \
            --loss_weight_csi 10.0 \
            --loss_weight_ssim 0.5 \
            --loss_weight_evo 2.0 \
            --loss_weight_spectral 2 \
            --early_stop_monitor val_score \
            --early_stop_mode max \
            --accelerator cuda \
            --devices 0,1\
            --precision bf16-mixed \
            --gradient_clip_val 5 \
            --gradient_clip_algorithm norm \
            --ckpt_path ./output/simvp/last.ckpt
        ;;
        
    # ============================================================
    # 2. 测试 SimVP 基座
    # ============================================================
    "test")
        echo "----------------------------------------"
        echo "🧪 开始测试 Mamba 基座模型..."
        echo "----------------------------------------"
        
        python run/test_scwds_simvp.py \
            --data_path data/samples.jsonl \
            --in_shape 10 54 256 256 \
            --aft_seq_length 20 \
            --save_dir ./output/simvp \
            --num_samples 10 \
            --accelerator cpu
        ;;
        
    # ============================================================
    # 3. 推理 SimVP 基座
    # ============================================================
    "infer")
        echo "----------------------------------------"
        echo "🔮 开始推理 Mamba 模型..."
        echo "----------------------------------------"
        
        python run/infer_scwds_simvp.py \
            --data_path data/samples.testset.jsonl \
            --in_shape 20 54 256 256 \
            --save_dir ./output/simvp \
            --accelerator cuda:0 \
            --vis
        ;;

    # ============================================================
    # 4. 推理 SimVP 基座 + Soft-GPM 后处理
    # ============================================================
    "infer_gpm")
        echo "----------------------------------------"
        echo "🔮 开始推理 SimVP (Soft-GPM) 模型..."
        echo "----------------------------------------"
        
        python run/infer_scwds_simvp_gpm.py \
            --data_path data/samples.testset.jsonl \
            --in_shape 20 54 256 256 \
            --save_dir ./output/simvp \
            --accelerator cuda:0 \
            --gpm_alpha 0.7 \
            --gpm_decay 0.9 \
            --vis
        ;;
        
    # ============================================================
    # 5. 推理 SimVP 基座 + Soft-FBC 后处理
    # ============================================================
    "infer_fbc")
        echo "----------------------------------------"
        echo "🔮 开始推理 SimVP (Soft-FBC) 模型..."
        echo "----------------------------------------"
        
        python run/infer_scwds_simvp_fbc.py \
            --data_path data/samples.testset.jsonl \
            --in_shape 20 54 256 256 \
            --save_dir ./output/simvp \
            --accelerator cuda:0 \
            --fbc_alpha 0.5 \
            --fbc_decay 0.9 \
            --ref_frames 10
        
esac

echo "✅ 操作完成！"