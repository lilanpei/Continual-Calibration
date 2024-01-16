python main.py -sn "JointTraining" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10" 
python main.py -sn "JointTraining" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10" -stcm
python main.py -sn "JointTraining" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10" -ppcm
python main.py -sn "Naive" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10"
python main.py -sn "Naive" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10" -stcm
python main.py -sn "Naive" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10" -ppcm
python main.py -sn "Naive" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10" -ppcm -ppdm
python main.py -sn "Replay" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10_ms_2000"
python main.py -sn "Replay" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10_ms_2000" -stcm
python main.py -sn "Replay" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10_ms_2000" -ppcm
python main.py -sn "Replay" -dn "EuroSAT" -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run1_epochs_100_ep_10_ms_2000" -ppcm -ppdm