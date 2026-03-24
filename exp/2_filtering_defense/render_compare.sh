GPU=0
NUM_IMAGES=100


SCENE_DATA="dataset/Nerf_Synthetic_adversarial/chair"
MODEL_PATH="log/10_filtering_defense/filtering_nerf_synthetic_epsilon_3.0/chair/exp_run_1"
PLY_PATH="${MODEL_PATH}/victim_model.ply"



echo "=============================================="
echo "Rendering ${NUM_IMAGES} comparison images (from PLY)"
echo "  Data:  ${SCENE_DATA}"
echo "  PLY:   ${PLY_PATH}"
echo "=============================================="

cd victim/gaussian-splatting

CUDA_VISIBLE_DEVICES=${GPU} python render_compare.py \
    -s "../../${SCENE_DATA}" \
    -m "../../${MODEL_PATH}" \
    --ply "../../${PLY_PATH}" \
    --num_images ${NUM_IMAGES}

cd ../..

echo ""
echo "Done! Results saved to: ${MODEL_PATH}/render_comparison/"