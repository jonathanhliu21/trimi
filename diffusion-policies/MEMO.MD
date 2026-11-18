.venv/bin/python ../trimorph-piper-control/inference/predictor/predictor.py   --policy-repo "$PWD"   --checkpoint "$PWD/data/data_trimorph_hand_mocap_quest_5/outputs/2026.11.04/01.08.24_train_diffusion_trimi_image_hand_cond_dinov2_16_trimi_image_hand_cond/checkpoints/latest.ckpt"   --device cuda:0   --port 5555


/home/romela5090/Jonathan/diffusion-policies/data/data_trimorph_hand_mocap_quest_5/outputs/2026.11.04/01.08.24_train_diffusion_trimi_image_hand_cond_dinov2_16_trimi_image_hand_cond/checkpoints/latest.ckpt

sudo tcpdump -ni wlo1 'host 192.168.20.16 and tcp port 5555'







TRAINING
CUDA_VISIBLE_DEVICES=0 python diffusion_policy/unet/train_unet_image_policy.py   --config-name=train_unet_image_trimi_hand_cond_dinov2_policy
