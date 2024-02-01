# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "JointTraining" 
# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "JointTraining" -stcm
# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "JointTraining" -ppcm
# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "Naive"
# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "Naive" -stcm
# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "Naive" -ppcm
# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "Naive" -ppcm -ppdm
# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "Replay"
# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "Replay" -stcm
# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "Replay" -ppcm
# python main.py --logdir "./logs/SplitMNIST_logs/run3_epochs_20_ep_3" -tp 20 -ep -sn "Replay" -ppcm -ppdm
echo "Launching runs..."
nohup python main.py -v 1 --cuda_id 1 --logdir ./logs/F_SplitMNIST -tp 50 -ep -p 5 -dn SplitMNIST -sn JointTraining > jointm_1.log 2>&1 &
nohup python main.py -v 2 --cuda_id 1 --logdir ./logs/F_SplitMNIST -tp 50 -ep -p 5 -dn SplitMNIST -sn JointTraining > jointm_2.log 2>&1 &
nohup python main.py -v 3 --cuda_id 2 --logdir ./logs/F_SplitMNIST -tp 50 -ep -p 5 -dn SplitMNIST -sn JointTraining > jointm_3.log 2>&1 &

nohup python main.py -v 1 --cuda_id 2 --logdir ./logs/F_SplitMNIST -tp 50 -ep -p 5 -dn SplitMNIST -sn JointTraining -stcm --ent_weight 0.075 > jointm_st1.log 2>&1 &
nohup python main.py -v 2 --cuda_id 3 --logdir ./logs/F_SplitMNIST -tp 50 -ep -p 5 -dn SplitMNIST -sn JointTraining -stcm --ent_weight 0.075 > jointm_st2.log 2>&1 &
nohup python main.py -v 3 --cuda_id 3 --logdir ./logs/F_SplitMNIST -tp 50 -ep -p 5 -dn SplitMNIST -sn JointTraining -stcm --ent_weight 0.075 > jointm_st3.log 2>&1 &