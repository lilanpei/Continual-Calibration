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
nohup python main.py --cuda_id 1 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" > run.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" -ew 1 -stcm > run1.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" -ew 0.1 -stcm > run2.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" -ew 0.075 -stcm > run3.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" -ew 0.05 -stcm > run4.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" -ew 0.025 -stcm > run5.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" -ew 0.01 -stcm > run6.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" -ew 0.0075 -stcm > run7.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" -ew 0.005 -stcm > run8.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" -ew 0.0025 -stcm > run9.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir "./logs/SplitMNIST_logs" -tp 50 -ep -p 5 -sn "JointTraining" -stcm > run10.log 2>&1 &