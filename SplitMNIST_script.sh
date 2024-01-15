python main.py -sn "JointTraining" -tp 50 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3" 
python main.py -sn "JointTraining" -tp 50 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3" -stcm
python main.py -sn "JointTraining" -tp 50 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3" -ppcm
python main.py -sn "Naive" -tp 20 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3"
python main.py -sn "Naive" -tp 20 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3" -stcm
python main.py -sn "Naive" -tp 20 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3" -ppcm
python main.py -sn "Naive" -tp 20 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3" -ppcm -ppdm
python main.py -sn "Replay" -tp 20 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3"
python main.py -sn "Replay" -tp 20 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3" -stcm
python main.py -sn "Replay" -tp 20 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3" -ppcm
python main.py -sn "Replay" -tp 20 --logdir "./logs/SplitMNIST_logs/run3_epochs_joint_50_CL_20_patience_3" -ppcm -ppdm