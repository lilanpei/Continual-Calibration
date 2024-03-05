# # nohup python main.py --logdir ./logs/run7 --cuda_id 1 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name Naive --dataset_name Atari > naive.log 2>&1 &
# # nohup python main.py --logdir ./logs/run7 --cuda_id 1 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name Naive --dataset_name Atari -stcm > naive_selftraining.log 2>&1 &
# # nohup python main.py --logdir ./logs/run7 --cuda_id 1 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name Naive --dataset_name Atari -ppcm > naive_postprocessing.log 2>&1 &
# # nohup python main.py --logdir ./logs/run7 --cuda_id 1 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name Naive --dataset_name Atari -ppcm -ppdm > naive_postprocessing_mixed.log 2>&1 &
# # nohup python main.py --logdir ./logs/run7 --cuda_id 2 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --mem_size 10000 --strategy_name Replay --dataset_name Atari > replay.log 2>&1 &
# # nohup python main.py --logdir ./logs/run7 --cuda_id 2 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --mem_size 10000 --strategy_name Replay --dataset_name Atari -stcm > replay_selftraining.log 2>&1 &
# # nohup python main.py --logdir ./logs/run7 --cuda_id 2 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --mem_size 10000 --strategy_name Replay --dataset_name Atari -ppcm > replay_postprocessing.log 2>&1 &
# # nohup python main.py --logdir ./logs/run7 --cuda_id 2 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --mem_size 10000 --strategy_name Replay --dataset_name Atari -ppcm -ppdm > replay_postprocessing_mixed.log 2>&1 &
# # nohup python main.py --logdir ./logs/run7 --cuda_id 3 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari > joint.log 2>&1 &
# # nohup python main.py --logdir ./logs/run7 --cuda_id 3 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm > joint_selftraining.log 2>&1 &
# # nohup python main.py --logdir ./logs/run7 --cuda_id 3 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -ppcm > joint_postprocessing.log 2>&1 &
# echo "Launching runs..."
# # nohup python main.py --logdir ./logs/run8 --cuda_id 0 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari > joint0.log 2>&1 &
# # nohup python main.py --logdir ./logs/run8 --cuda_id 0 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm --ent_weight 1 > joint1.log 2>&1 &
# # nohup python main.py --logdir ./logs/run8 --cuda_id 0 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm --ent_weight 0.1 > joint2.log 2>&1 &
# # nohup python main.py --logdir ./logs/run8 --cuda_id 0 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm --ent_weight 0.075 > joint3.log 2>&1 &
# # nohup python main.py --logdir ./logs/run8 --cuda_id 1 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm --ent_weight 0.05 > joint4.log 2>&1 &
# # nohup python main.py --logdir ./logs/run8 --cuda_id 1 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm --ent_weight 0.025 > joint5.log 2>&1 &
# # nohup python main.py --logdir ./logs/run8 --cuda_id 1 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm --ent_weight 0.01 > joint6.log 2>&1 &
# # nohup python main.py --logdir ./logs/run8 --cuda_id 2 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm --ent_weight 0.0075 > joint7.log 2>&1 &
# # nohup python main.py --logdir ./logs/run8 --cuda_id 2 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm --ent_weight 0.005 > joint8.log 2>&1 &
# # nohup python main.py --logdir ./logs/run8 --cuda_id 2 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm --ent_weight 0.0025 > joint9.log 2>&1 &
# # nohup python main.py --logdir ./logs/run8 --cuda_id 2 --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name JointTraining --dataset_name Atari -stcm > joint_st.log 2>&1 &

# -----------

# nohup python main.py -v 1 --cuda_id 0 --logdir ./logs/Atari -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 -dn Atari -sn JointTraining > jointa_1.log 2>&1 &
# nohup python main.py -v 2 --cuda_id 1 --logdir ./logs/Atari -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 -dn Atari -sn JointTraining > jointa_2.log 2>&1 &
# nohup python main.py -v 3 --cuda_id 2 --logdir ./logs/Atari -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 -dn Atari -sn JointTraining > jointa_3.log 2>&1 &

# nohup python main.py -v 1 --cuda_id 3 --logdir ./logs/Atari_ST -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 -dn Atari -sn JointTraining -stcm --ent_weight 0.0075 > jointa_st1.log 2>&1 &
# nohup python main.py -v 2 --cuda_id 1 --logdir ./logs/Atari_ST -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 -dn Atari -sn JointTraining -stcm --ent_weight 0.0075 > jointa_st2.log 2>&1 &
# nohup python main.py -v 3 --cuda_id 2 --logdir ./logs/Atari_ST -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 -dn Atari -sn JointTraining -stcm --ent_weight 0.0075 > jointa_st3.log 2>&1 &

# nohup python main.py -sn "JointTraining" -dn "Atari" -v 1 -cid 0 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS" -lrpp 0.01 -mi 100 -ppcm -ppms > jointcms_1.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 2 -cid 0 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS" -lrpp 0.01 -mi 100 -ppcm -ppms > jointcms_2.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 3 -cid 0 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS" -lrpp 0.01 -mi 100 -ppcm -ppms > jointcms_3.log 2>&1 &

# nohup python main.py -sn "JointTraining" -dn "Atari" -v 1 -cid 1 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS" -lrpp 0.01 -mi 100 -ppcm -ppvs > jointcvs_1.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 2 -cid 1 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS" -lrpp 0.01 -mi 100 -ppcm -ppvs > jointcvs_2.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 3 -cid 1 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS" -lrpp 0.01 -mi 100 -ppcm -ppvs > jointcvs_3.log 2>&1 &

# nohup python main.py -sn "JointTraining" -dn "Atari" -v 1 -cid 2 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS" -lrpp 0.01 -mi 100 -ppcm > jointcts_1.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 2 -cid 2 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS" -lrpp 0.01 -mi 100 -ppcm > jointcts_2.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 3 -cid 2 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS" -lrpp 0.01 -mi 100 -ppcm > jointcts_3.log 2>&1 &

# nohup python main.py -sn "JointTraining" -dn "Atari" -v 1 -cid 3 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS2" -lrpp 0.001 -mi 100 -ppcm -ppms > jointcms2_1.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 2 -cid 3 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS2" -lrpp 0.001 -mi 100 -ppcm -ppms > jointcms2_2.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 3 -cid 3 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS2" -lrpp 0.001 -mi 100 -ppcm -ppms > jointcms2_3.log 2>&1 &

# nohup python main.py -sn "JointTraining" -dn "Atari" -v 1 -cid 0 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS2" -lrpp 0.001 -mi 100 -ppcm -ppvs > jointcvs2_1.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 2 -cid 1 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS2" -lrpp 0.001 -mi 100 -ppcm -ppvs > jointcvs2_2.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 3 -cid 2 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS2" -lrpp 0.001 -mi 100 -ppcm -ppvs > jointcvs2_3.log 2>&1 &

# nohup python main.py -sn "JointTraining" -dn "Atari" -v 1 -cid 3 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS2" -lrpp 0.001 -mi 100 -ppcm > jointcts2_1.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 2 -cid 2 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS2" -lrpp 0.001 -mi 100 -ppcm > jointcts2_2.log 2>&1 &
# nohup python main.py -sn "JointTraining" -dn "Atari" -v 3 -cid 3 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS2" -lrpp 0.001 -mi 100 -ppcm > jointcts2_3.log 2>&1 &

# -----------

# nohup python main.py -v 1 -cid 0 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R" --mem_size 4000 > ar1.log 2>&1 &
# nohup python main.py -v 2 -cid 0 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R" --mem_size 4000 > ar2.log 2>&1 &
# nohup python main.py -v 3 -cid 0 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R" --mem_size 4000 > ar3.log 2>&1 &

# nohup python main.py -v 2 -cid 1 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_ST" --mem_size 4000 -stcm --ent_weight 0.0025 > r1.log 2>&1 &
# nohup python main.py -v 3 -cid 1 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_ST" --mem_size 4000 -stcm --ent_weight 0.0025 > r2.log 2>&1 &
# nohup python main.py -v 4 -cid 1 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_ST" --mem_size 4000 -stcm --ent_weight 0.0025 > r3.log 2>&1 &

# nohup python main.py -v 1 -cid 2 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppms > jointcms_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppms > jointcms_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppms > jointcms_3.log 2>&1 &

# nohup python main.py -v 1 -cid 3 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppvs > jointcvs_1.log 2>&1 &
# nohup python main.py -v 2 -cid 3 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppvs > jointcvs_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppvs > jointcvs_3.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm > jointcts_1.log 2>&1 &
# nohup python main.py -v 2 -cid 0 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm > jointcts_2.log 2>&1 &
# nohup python main.py -v 3 -cid 0 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm > jointcts_3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS2" --mem_size 4000 -lrpp 0.001 -mi 100 -ppcm -ppms > jointcms2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS2" --mem_size 4000 -lrpp 0.001 -mi 100 -ppcm -ppms > jointcms2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 1 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS2" --mem_size 4000 -lrpp 0.001 -mi 100 -ppcm -ppms > jointcms2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 2 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS2" --mem_size 4000 -lrpp 0.001 -mi 100 -ppcm -ppvs > jointcvs2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS2" --mem_size 4000 -lrpp 0.001 -mi 100 -ppcm -ppvs > jointcvs2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS2" --mem_size 4000 -lrpp 0.001 -mi 100 -ppcm -ppvs > jointcvs2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 3 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS2" --mem_size 4000 -lrpp 0.001 -mi 100 -ppcm > jointcts2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 3 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS2" --mem_size 4000 -lrpp 0.001 -mi 100 -ppcm > jointcts2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS2" --mem_size 4000 -lrpp 0.001 -mi 100 -ppcm > jointcts2_3.log 2>&1 &

# -----------

# nohup python main.py -v 1 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N" > ar1.log 2>&1 &
# nohup python main.py -v 2 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N" > ar2.log 2>&1 &
# nohup python main.py -v 3 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N" > ar3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_ST" -stcm --ent_weight 0.01 > n1.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_ST" -stcm --ent_weight 0.01 > n2.log 2>&1 &
# nohup python main.py -v 3 -cid 1 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_ST" -stcm --ent_weight 0.01 > n3.log 2>&1 &

# nohup python main.py -v 1 -cid 2 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_ST" -stcm --ent_weight 0.005 > n4.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_ST" -stcm --ent_weight 0.005 > n5.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_ST" -stcm --ent_weight 0.005 > n6.log 2>&1 &

# nohup python main.py -v 1 -cid 3 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_MS" -lrpp 0.01 -mi 100 -ppcm -ppms > naivecms_1.log 2>&1 &
# nohup python main.py -v 2 -cid 3 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_MS" -lrpp 0.01 -mi 100 -ppcm -ppms > naivecms_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_MS" -lrpp 0.01 -mi 100 -ppcm -ppms > naivecms_3.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_VS" -lrpp 0.01 -mi 100 -ppcm -ppvs > naivecvs_1.log 2>&1 &
# nohup python main.py -v 2 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_VS" -lrpp 0.01 -mi 100 -ppcm -ppvs > naivecvs_2.log 2>&1 &
# nohup python main.py -v 3 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_VS" -lrpp 0.01 -mi 100 -ppcm -ppvs > naivecvs_3.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_TS" -lrpp 0.01 -mi 100 -ppcm > naivects_1.log 2>&1 &
# nohup python main.py -v 2 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_TS" -lrpp 0.01 -mi 100 -ppcm > naivects_2.log 2>&1 &
# nohup python main.py -v 3 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_TS" -lrpp 0.01 -mi 100 -ppcm > naivects_3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_MS2" -lrpp 0.001 -mi 100 -ppcm -ppms > naivecms2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_MS2" -lrpp 0.001 -mi 100 -ppcm -ppms > naivecms2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 1 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_MS2" -lrpp 0.001 -mi 100 -ppcm -ppms > naivecms2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 2 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_VS2" -lrpp 0.001 -mi 100 -ppcm -ppvs > naivecvs2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_VS2" -lrpp 0.001 -mi 100 -ppcm -ppvs > naivecvs2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_VS2" -lrpp 0.001 -mi 100 -ppcm -ppvs > naivecvs2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 3 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_TS2" -lrpp 0.001 -mi 100 -ppcm > naivects2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 3 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_TS2" -lrpp 0.001 -mi 100 -ppcm > naivects2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_TS2" -lrpp 0.001 -mi 100 -ppcm > naivects2_3.log 2>&1 &

# -----------

# nohup python main.py -v 1 -cid 0 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS_MD" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppms > jointcms_md_1.log 2>&1 &
# nohup python main.py -v 2 -cid 0 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS_MD" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppms > jointcms_md_2.log 2>&1 &
# nohup python main.py -v 3 -cid 0 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_MS_MD" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppms > jointcms_md_3.log 2>&1 &

# nohup python main.py -v 1 -cid 2 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS_MD" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppvs > jointcvs_md_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS_MD" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppvs > jointcvs_md_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_VS_MD" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppvs > jointcvs_md_3.log 2>&1 &

# nohup python main.py -v 1 -cid 3 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS_MD" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppdm > jointcts_md_1.log 2>&1 &
# nohup python main.py -v 2 -cid 3 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS_MD" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppdm > jointcts_md_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "Replay" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_R_TS_MD" --mem_size 4000 -lrpp 0.01 -mi 100 -ppcm -ppdm > jointcts_md_3.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_TS_MD" -lrpp 0.001 -mi 100 -ppcm -ppdm > jointcts2_md_1.log 2>&1 &
# nohup python main.py -v 2 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_TS_MD" -lrpp 0.001 -mi 100 -ppcm -ppdm > jointcts2_md_2.log 2>&1 &
# nohup python main.py -v 3 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_TS_MD" -lrpp 0.001 -mi 100 -ppcm -ppdm > jointcts2_md_3.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_VS_MD" -lrpp 0.001 -mi 100 -ppcm -ppdm -ppvs > jointcvs2_md_1.log 2>&1 &
# nohup python main.py -v 2 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_VS_MD" -lrpp 0.001 -mi 100 -ppcm -ppdm -ppvs > jointcvs2_md_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_VS_MD" -lrpp 0.001 -mi 100 -ppcm -ppdm -ppvs > jointcvs2_md_3.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_MS_MD" -lrpp 0.01 -mi 100 -ppcm -ppdm -ppms > jointcms_md_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_MS_MD" -lrpp 0.01 -mi 100 -ppcm -ppdm -ppms > jointcms_md_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "Naive" -dn "Atari" -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_N_MS_MD" -lrpp 0.01 -mi 100 -ppcm -ppdm -ppms > jointcms_md_3.log 2>&1 &

# -----------

# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.1 -b 0.5 -bsm 32 > d4.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.1 -b 0.5 -bsm 32 > d5.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.1 -b 0.5 -bsm 32 > d6.log 2>&1 &

# nohup python main.py -v 4 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.1 -b 0.8 -bsm 32 > d44.log 2>&1 &
# nohup python main.py -v 5 -cid 0 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.1 -b 0.8 -bsm 32 > d55.log 2>&1 &
# nohup python main.py -v 6 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.1 -b 0.8 -bsm 32 > d66.log 2>&1 &

# nohup python main.py -v 7 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.2 -b 0.5 -bsm 32 > d4.log 2>&1 &
# nohup python main.py -v 8 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.2 -b 0.5 -bsm 32 > d5.log 2>&1 &
# nohup python main.py -v 9 -cid 0 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.2 -b 0.5 -bsm 32 > d6.log 2>&1 &

# nohup python main.py -v 10 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.2 -b 0.8 -bsm 32 > d44.log 2>&1 &
# nohup python main.py -v 11 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.2 -b 0.8 -bsm 32 > d55.log 2>&1 &
# nohup python main.py -v 12 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.2 -b 0.8 -bsm 32 > d66.log 2>&1 &

# nohup python main.py -v 13 -cid 0 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.3 -b 0.5 -bsm 32 > d444.log 2>&1 &
# nohup python main.py -v 14 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.3 -b 0.5 -bsm 32 > d555.log 2>&1 &
# nohup python main.py -v 15 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.3 -b 0.5 -bsm 32 > d666.log 2>&1 &

# nohup python main.py -v 16 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.3 -b 0.8 -bsm 32 > d4444.log 2>&1 &
# nohup python main.py -v 17 -cid 0 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.3 -b 0.8 -bsm 32 > d5555.log 2>&1 &
# nohup python main.py -v 18 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.3 -b 0.8 -bsm 32 > d6666.log 2>&1 &

# nohup python main.py -v 19 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.5 -b 0.5 -bsm 32 > d44444.log 2>&1 &
# nohup python main.py -v 20 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.5 -b 0.5 -bsm 32 > d55555.log 2>&1 &
# nohup python main.py -v 21 -cid 0 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.5 -b 0.5 -bsm 32 > d66666.log 2>&1 &

# nohup python main.py -v 22 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.5 -b 0.8 -bsm 32 > d444444.log 2>&1 &
# nohup python main.py -v 23 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.5 -b 0.8 -bsm 32 > d555555.log 2>&1 &
# nohup python main.py -v 24 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 2000 -a 0.5 -b 0.8 -bsm 32 > d666666.log 2>&1 &

# nohup python main.py -v 25 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 4000 -a 0.5 -b 0.5 -bsm 32 > adb1.log 2>&1 &
# nohup python main.py -v 26 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 4000 -a 0.5 -b 0.5 -bsm 32 > adb11.log 2>&1 &
# nohup python main.py -v 27 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_HP" -ms 4000 -a 0.5 -b 0.5 -bsm 32 > adb111.log 2>&1 &

# -------

# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.1 > ad1.log 2>&1 &
# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.075 > ad2.log 2>&1 &
# nohup python main.py -v 1 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.05 > ad3.log 2>&1 &
# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.025 > ad4.log 2>&1 &
# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.01 > ad5.log 2>&1 &
# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.0075 > ad6.log 2>&1 &
# nohup python main.py -v 1 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.005 > ad7.log 2>&1 &
# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.0025 > ad8.log 2>&1 &

# -------

# nohup python main.py -v 2 -cid 0 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.025 > ad4.log 2>&1 &
# nohup python main.py -v 3 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.025 > ad4.log 2>&1 &

# nohup python main.py -v 2 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.0025 > ad8.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_ST" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -stcm --ent_weight 0.0025 > ad8.log 2>&1 &

# -------

# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_MS" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppms > adms_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_MS" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppms > adms_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_MS" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppms > adms_3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_VS" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppvs > advs_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_VS" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppvs > advs_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_VS" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppvs > advs_3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_TS" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm > adts_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_TS" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm > adts_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_TS" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm > adts_3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_MS2" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppms > adms2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_MS2" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppms > adms2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_MS2" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppms > adms2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_VS2" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppvs > advs2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_VS2" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppvs > advs2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_VS2" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppvs > advs2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_TS2" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.001 -mi 100 -ppcm > adts2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_TS2" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.001 -mi 100 -ppcm > adts2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_TS2" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.001 -mi 100 -ppcm > adts2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_MS_MD" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppms > adms2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_MS_MD" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppms > adms2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_MS_MD" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppms > adms2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_VS_MD" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppvs > advs2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_VS_MD" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppvs > advs2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_VS_MD" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppdm -ppvs > advs2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_TS_MD" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppdm > adts2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_TS_MD" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppdm > adts2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 1 -sn "DER" -dn "Atari" -ts 256 -es 256 -tp 100 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_D_TS_MD" -ms 2000 -a 0.5 -b 0.5 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppdm > adts2_3.log 2>&1 &