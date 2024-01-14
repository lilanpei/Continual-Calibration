python main.py -sn "JointTraining" -tp 25 --logdir "./logs/run1" 
python main.py -sn "JointTraining" -tp 25 --logdir "./logs/run1" -stcm
python main.py -sn "JointTraining" -tp 25 --logdir "./logs/run1" -ppcm
python main.py -sn "Naive" -tp 5 --logdir "./logs/run1"
python main.py -sn "Naive" -tp 5 --logdir "./logs/run1" -stcm
python main.py -sn "Naive" -tp 5 --logdir "./logs/run1" -ppcm
python main.py -sn "Naive" -tp 5 --logdir "./logs/run1" -ppcm -ppdm
python main.py -sn "Replay" -tp 5 --logdir "./logs/run1"
python main.py -sn "Replay" -tp 5 --logdir "./logs/run1" -stcm
python main.py -sn "Replay" -tp 5 --logdir "./logs/run1" -ppcm
python main.py -sn "Replay" -tp 5 --logdir "./logs/run1" -ppcm -ppdm