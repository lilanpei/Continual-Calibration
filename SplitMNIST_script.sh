python main.py -sn "JointTraining" -tp 50 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20" 
python main.py -sn "JointTraining" -tp 50 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20" -stcm
python main.py -sn "JointTraining" -tp 50 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20" -ppcm
python main.py -sn "Naive" -tp 20 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20"
python main.py -sn "Naive" -tp 20 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20" -stcm
python main.py -sn "Naive" -tp 20 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20" -ppcm
python main.py -sn "Naive" -tp 20 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20" -ppcm -ppdm
python main.py -sn "Replay" -tp 20 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20"
python main.py -sn "Replay" -tp 20 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20" -stcm
python main.py -sn "Replay" -tp 20 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20" -ppcm
python main.py -sn "Replay" -tp 20 --logdir "./logs/SplitMNIST_logs/run2_epochs_joint_50_CL_20" -ppcm -ppdm