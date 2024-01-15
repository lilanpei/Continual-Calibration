python main.py -sn "JointTraining" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20" 
python main.py -sn "JointTraining" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20" -stcm
python main.py -sn "JointTraining" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20" -ppcm
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20"
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20" -stcm
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20" -ppcm
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20" -ppcm -ppdm
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20"
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20" -stcm
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20" -ppcm
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/SplitCIFAR100_logs/run1_epochs_joint_200_CL_20" -ppcm -ppdm