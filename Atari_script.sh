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

# nohup python main.py -v 1 --cuda_id 1 --logdir ./logs/F_Atari3 -ts 256 -es 256 -tp 200 -lr 0.005 -ep -p 10 -dn Atari -sn JointTraining > jointa_1.log 2>&1 &
# nohup python main.py -v 2 --cuda_id 1 --logdir ./logs/F_Atari3 -ts 256 -es 256 -tp 200 -lr 0.005 -ep -p 10 -dn Atari -sn JointTraining > jointa_2.log 2>&1 &
# nohup python main.py -v 3 --cuda_id 2 --logdir ./logs/F_Atari3 -ts 256 -es 256 -tp 200 -lr 0.005 -ep -p 10 -dn Atari -sn JointTraining > jointa_3.log 2>&1 &
# nohup python main.py -v 1 --cuda_id 2 --logdir ./logs/F_Atari3 -ts 256 -es 256 -tp 200 -lr 0.005 -ep -p 10 -dn Atari -sn JointTraining -stcm --ent_weight 0.0075 > jointa_st1.log 2>&1 &
# nohup python main.py -v 2 --cuda_id 3 --logdir ./logs/F_Atari3 -ts 256 -es 256 -tp 200 -lr 0.005 -ep -p 10 -dn Atari -sn JointTraining -stcm --ent_weight 0.0075 > jointa_st2.log 2>&1 &
# nohup python main.py -v 3 --cuda_id 3 --logdir ./logs/F_Atari3 -ts 256 -es 256 -tp 200 -lr 0.005 -ep -p 10 -dn Atari -sn JointTraining -stcm --ent_weight 0.0075 > jointa_st3.log 2>&1 &

nohup python main.py -sn "JointTraining" -dn "Atari" -v 1 -cid 1 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat1" -lrpp 0.3 -mi 500 -ppcm -ppms > jointcma_1.log 2>&1 &
nohup python main.py -sn "JointTraining" -dn "Atari" -v 2 -cid 3 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat1" -lrpp 0.3 -mi 500 -ppcm -ppms > jointcma_2.log 2>&1 &
nohup python main.py -sn "JointTraining" -dn "Atari" -v 3 -cid 2 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat1" -lrpp 0.3 -mi 500 -ppcm -ppms > jointcma_3.log 2>&1 &

nohup python main.py -sn "JointTraining" -dn "Atari" -v 1 -cid 1 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat2" -lrpp 0.01 -mi 500 -ppcm -ppms > jointcma_11.log 2>&1 &
nohup python main.py -sn "JointTraining" -dn "Atari" -v 2 -cid 3 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat2" -lrpp 0.01 -mi 500 -ppcm -ppms > jointcma_22.log 2>&1 &
nohup python main.py -sn "JointTraining" -dn "Atari" -v 3 -cid 2 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat2" -lrpp 0.01 -mi 500 -ppcm -ppms > jointcma_33.log 2>&1 &

nohup python main.py -sn "JointTraining" -dn "Atari" -v 1 -cid 2 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat1" -lrpp 0.3 -mi 500 -ppcm -ppvs > jointcva_1.log 2>&1 &
nohup python main.py -sn "JointTraining" -dn "Atari" -v 2 -cid 1 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat1" -lrpp 0.3 -mi 500 -ppcm -ppvs > jointcva_2.log 2>&1 &
nohup python main.py -sn "JointTraining" -dn "Atari" -v 3 -cid 3 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat1" -lrpp 0.3 -mi 500 -ppcm -ppvs > jointcva_3.log 2>&1 &

nohup python main.py -sn "JointTraining" -dn "Atari" -v 1 -cid 2 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat2" -lrpp 0.01 -mi 500 -ppcm -ppvs > jointcva_11.log 2>&1 &
nohup python main.py -sn "JointTraining" -dn "Atari" -v 2 -cid 1 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat2" -lrpp 0.01 -mi 500 -ppcm -ppvs > jointcva_22.log 2>&1 &
nohup python main.py -sn "JointTraining" -dn "Atari" -v 3 -cid 3 -ts 256 -es 256 -tp 200 -lr 0.0005 -ep -p 10 --logdir "./logs/Atari_vetmat2" -lrpp 0.01 -mi 500 -ppcm -ppvs > jointcva_33.log 2>&1 &