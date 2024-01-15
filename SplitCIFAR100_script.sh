python main.py -sn "JointTraining" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 --logdir "./logs/run1" 
python main.py -sn "JointTraining" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 --logdir "./logs/run1" -stcm
python main.py -sn "JointTraining" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 200 -lr 0.1 --logdir "./logs/run1" -ppcm
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/run1"
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/run1" -stcm
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/run1" -ppcm
python main.py -sn "Naive" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/run1" -ppcm -ppdm
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/run1"
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/run1" -stcm
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/run1" -ppcm
python main.py -sn "Replay" -dn "SplitCIFAR100" -ts 128 -es 128 -tp 20 -lr 0.1 --logdir "./logs/run1" -ppcm -ppdm