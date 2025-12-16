#!/bin/bash
# SCWDS Nowcasting 全流程脚本
# 包含: Train -> Test -> Infer  ->
# Usage: bash run.scwds.mamba.sh [MODE]

# ================= 环境变量优化 =================
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# ================= 参数检查 =================
if [ $# -eq 0 ]; then
    echo "错误: 请指定操作模式"
    echo "用法: bash run.scwds.mamba.sh [MODE]"
    echo "支持的模式:"
    echo " train      - 训练 Mamba 基座模型"
    echo " test       - 测试 Mamba 基座模型"
    echo " infer      - 使用 Mamba 基座进行推理"
    exit 1
fi

MODE=$1

case $MODE in
    # ============================================================
    # 1. 训练 Mamba 基座
    # ============================================================
    "train")
        echo "--------------------------------------------------------"
        echo "🚀 开始训练 Mamba 基座模型 (BF16 Mixed)..."
        echo "--------------------------------------------------------"
        
        python run/train_scwds_mamba.py fit \
        --seed_everything 42 \
        \
        --trainer.default_root_dir "./output/mamba" \
        --trainer.accelerator cuda \
        --trainer.devices 0,1,2,3 \
        --trainer.strategy ddp \
        --trainer.precision 16-mixed \
        --trainer.max_epochs 100 \
        --trainer.log_every_n_steps 100 \
        --trainer.accumulate_grad_batches 4 \
        --trainer.gradient_clip_val 0.5 \
        --trainer.gradient_clip_algorithm "norm" \
        \
        --trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint \
        --trainer.callbacks.dirpath "./output/mamba/checkpoints" \
        --trainer.callbacks.monitor "val_score" \
        --trainer.callbacks.mode "max" \
        --trainer.callbacks.save_top_k "-1" \
        --trainer.callbacks.save_last true \
        --trainer.callbacks.filename "{epoch:02d}-{val_score:.4f}" \
        \
        --trainer.callbacks+=lightning.pytorch.callbacks.EarlyStopping \
        --trainer.callbacks.monitor "val_score" \
        --trainer.callbacks.mode "max" \
        --trainer.callbacks.patience 20 \
        \
        --data.data_path "data/samples.jsonl" \
        --data.batch_size 1 \
        --data.num_workers 8 \
        \
        --model.in_shape "[10, 54, 256, 256]" \
        --model.out_seq_length 20 \
        --model.hid_S 128 \
        --model.hid_T 512 \
        --model.N_S 4 \
        --model.N_T 12 \
        --model.mlp_ratio 4.0 \
        --model.spatio_kernel_enc 5 \
        --model.spatio_kernel_dec 5 \
        --model.drop 0.0 \
        --model.drop_path 0.1 \
        \
        --model.d_state 16 \
        --model.d_conv 4 \
        --model.expand 2 \
        \
        --model.lr 5e-5 \
        --model.opt "adamw" \
        --model.weight_decay 0.05 \
        --model.filter_bias_and_bn true \
        --model.sched "cosine" \
        --model.min_lr 1e-6 \
        --model.warmup_lr 1e-6 \
        --model.warmup_epoch 10 \
        --model.decay_epoch 30 \
        --model.decay_rate 0.1 \
        --model.use_curriculum_learning true \
        \
        --model.loss_weight_l1 1.0 \
        --model.loss_weight_ssim 0.5 \
        --model.loss_weight_csi 1.0 \
        --model.loss_weight_spectral 0.5 \
        --model.loss_weight_evo 0.5 \
        --ckpt_path ./output/mamba/checkpoints/last.ckpt \
        ;;
        
    # ============================================================
    # 2. 测试
    # ============================================================
    "test")
        echo "----------------------------------------"
        echo "🧪 开始测试 Mamba 基座模型..."
        echo "----------------------------------------"
        
        python run/test_scwds_mamba.py \
            --data_path data/samples.jsonl \
            --in_shape 10 54 256 256 \
            --out_seq_length 20 \
            --save_dir ./output/mamba \
            --num_samples 10 \
            --accelerator cuda \
            # --ckpt_path ./output/mamba/checkpoints/epoch=16-val_score=0.0488.ckpt
        ;;
        
    # ============================================================
    # 3. 推理
    # ============================================================
    "infer")
        echo "----------------------------------------"
        echo "🔮 开始推理 Mamba 模型..."
        echo "----------------------------------------"
        
        python run/infer_scwds_mamba.py \
            --data_path data/samples.testset.jsonl \
            --in_shape 20 54 256 256 \
            --save_dir ./output/mamba \
            --accelerator cuda
        ;;
       
esac

echo "✅ 操作完成！"