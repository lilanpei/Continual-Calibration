# python main.py -sn "JointTraining" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" 
# python main.py -sn "JointTraining" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -stcm
# python main.py -sn "JointTraining" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm
# python main.py -sn "JointTraining" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppvs
# python main.py -sn "JointTraining" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppms
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3"
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -stcm
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppdm
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppvs
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppvs -ppdm
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppms
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppms -ppdm
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3"
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -stcm
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppdm
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppvs
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppvs -ppdm
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppms
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppms -ppdm

# nohup python main.py -v 1 -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" > r1.log 2>&1 &
# nohup python main.py -v 2 -sn "Replay" -cid 2 -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" > r2.log 2>&1 &
# nohup python main.py -v 3 -sn "Replay" -cid 3 -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" > r3.log 2>&1 &
# nohup python main.py -v 1 -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" > n1.log 2>&1 &
# nohup python main.py -v 2 -sn "Naive" -cid 2 -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" > n2.log 2>&1 &
# nohup python main.py -v 3 -sn "Naive" -cid 3 -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" > n3.log 2>&1 &

# nohup python main.py -cid 0 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.1 > r1.log 2>&1 &
# nohup python main.py -cid 1 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.075 > r2.log 2>&1 &
# nohup python main.py -cid 2 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.05 > r3.log 2>&1 &
# nohup python main.py -cid 3 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.025 > r4.log 2>&1 &
# nohup python main.py -cid 0 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.01 > r5.log 2>&1 &
# nohup python main.py -cid 1 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.0075 > r6.log 2>&1 &
# nohup python main.py -cid 2 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.005 > r7.log 2>&1 &
# nohup python main.py -cid 3 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.0025 > r8.log 2>&1 &

# nohup python main.py -cid 0 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.1 > n1.log 2>&1 &
# nohup python main.py -cid 1 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.075 > n2.log 2>&1 &
# nohup python main.py -cid 2 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.05 > n3.log 2>&1 &
# nohup python main.py -cid 3 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.025 > n4.log 2>&1 &
# nohup python main.py -cid 0 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.01 > n5.log 2>&1 &
# nohup python main.py -cid 1 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.0075 > n6.log 2>&1 &
# nohup python main.py -cid 2 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.005 > n7.log 2>&1 &
# nohup python main.py -cid 3 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.0025 > n8.log 2>&1 &

nohup python main.py -v 2 -cid 0 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.025 > rm1.log 2>&1 &
nohup python main.py -v 3 -cid 0 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.025 > rm2.log 2>&1 &
nohup python main.py -v 4 -cid 0 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.025 > rm3.log 2>&1 &

nohup python main.py -v 2 -cid 2 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.05 > rm4.log 2>&1 &
nohup python main.py -v 3 -cid 2 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.05 > rm5.log 2>&1 &
nohup python main.py -v 4 -cid 2 -sn "Replay" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Replay" -stcm --ent_weight 0.05 > rm6.log 2>&1 &

nohup python main.py -v 2 -cid 1 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.075 > nm1.log 2>&1 &
nohup python main.py -v 3 -cid 1 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.075 > nm1.log 2>&1 &
nohup python main.py -v 4 -cid 1 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.075 > nm3.log 2>&1 &

nohup python main.py -v 2 -cid 3 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.0025 > nm4.log 2>&1 &
nohup python main.py -v 3 -cid 3 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.0025 > nm5.log 2>&1 &
nohup python main.py -v 4 -cid 3 -sn "Naive" -dn "SplitMNIST" -tp 20 -ep --logdir "./logs/SplitMNIST_Naive" -stcm --ent_weight 0.0025 > nm6.log 2>&1 &



