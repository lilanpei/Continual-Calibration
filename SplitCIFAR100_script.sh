python main.py -sn "JointTraining" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10" 
python main.py -sn "JointTraining" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10" -stcm
python main.py -sn "JointTraining" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10" -ppcm
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10"
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10" -stcm
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10" -ppcm
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10" -ppcm -ppdm
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ms 4000 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10"
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ms 4000 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10" -stcm
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ms 4000 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10" -ppcm
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 -ms 4000 -ep -p 10 --logdir "./logs/SplitCIFAR100_logs/run2_epochs_200_ep_10" -ppcm -ppdm