function infer_mix()
{
    python tests/inference_edit.py \
        --ckpt /vePFS-buaa/yunweibin/Unpaired/MixUniLat/lightning_ckpts/step=step=18000.ckpt \
        --config /vePFS-buaa/yunweibin/Unpaired/MixUniLat/config.json \
        --batch_file /vePFS-buaa/yunweibin/workspace/UniLat3D/outputs/infer_list.json \
        --output_dir ./outputs/inference_results_mix_18000 \
        --device cuda \
        --export_mesh \
        --pretrained_dir /vePFS-buaa/yunweibin/workspace/UniLat3D/pretrained \
        --eval_views
}

function infer_unilat()
{
    python tests/unilat_inference.py \
        --batch_file /vePFS-buaa/yunweibin/workspace/UniLat3D/samples/selected.json \
        --output_dir ./samples/outputs_unilat \
        --device cuda \
        --export_mesh \
        --pretrained_dir /vePFS-buaa/yunweibin/workspace/UniLat3D/pretrained \
        --num_steps 20 \
        --cfg_strength 7.5 \
        --eval_views
}

CUDA_VISIBLE_DEVICES=6 infer_unilat