# python main.py -sn "JointTraining" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" 
# python main.py -sn "JointTraining" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -stcm
# python main.py -sn "JointTraining" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm
# python main.py -sn "JointTraining" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppvs
# python main.py -sn "JointTraining" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppms
# python main.py -sn "Naive" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100"
# python main.py -sn "Naive" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -stcm
# python main.py -sn "Naive" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm
# python main.py -sn "Naive" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppdm
# python main.py -sn "Naive" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppvs
# python main.py -sn "Naive" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppvs -ppdm
# python main.py -sn "Naive" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppms
# python main.py -sn "Naive" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppms -ppdm
# python main.py -sn "Replay" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100"
# python main.py -sn "Replay" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -stcm
# python main.py -sn "Replay" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm
# python main.py -sn "Replay" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppdm
# python main.py -sn "Replay" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppvs
# python main.py -sn "Replay" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppvs -ppdm
# python main.py -sn "Replay" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppms
# python main.py -sn "Replay" -dn "EuroSAT" -cid 1 -ts 128 -es 128 -tp 100 -lr 0.01 -ms 2000 -ep -p 10 --logdir "./logs/EuroSAT_logs/run7_epochs_100" -ppcm -ppms -ppdm

# ------------------

# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_500" -ms 500 -a 0.1 -b 0.5 -bsm 32 > d1.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_500" -ms 500 -a 0.1 -b 0.5 -bsm 32 > d2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_500" -ms 500 -a 0.1 -b 0.5 -bsm 32 > d3.log 2>&1 &

# nohup python main.py -v 4 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_500" -ms 500 -a 0.1 -b 0.8 -bsm 32 > d11.log 2>&1 &
# nohup python main.py -v 5 -cid 0 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_500" -ms 500 -a 0.1 -b 0.8 -bsm 32 > d22.log 2>&1 &
# nohup python main.py -v 6 -cid 1 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_500" -ms 500 -a 0.1 -b 0.8 -bsm 32 > d33.log 2>&1 &

# nohup python main.py -v 1 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.1 -b 0.5 -bsm 32 > d4.log 2>&1 &
# nohup python main.py -v 2 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.1 -b 0.5 -bsm 32 > d5.log 2>&1 &
# nohup python main.py -v 3 -cid 0 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.1 -b 0.5 -bsm 32 > d6.log 2>&1 &

# nohup python main.py -v 4 -cid 1 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.1 -b 0.8 -bsm 32 > d44.log 2>&1 &
# nohup python main.py -v 5 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.1 -b 0.8 -bsm 32 > d55.log 2>&1 &
# nohup python main.py -v 6 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.1 -b 0.8 -bsm 32 > d66.log 2>&1 &

# nohup python main.py -v 7 -cid 0 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.2 -b 0.5 -bsm 32 > d4.log 2>&1 &
# nohup python main.py -v 8 -cid 0 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.2 -b 0.5 -bsm 32 > d5.log 2>&1 &
# nohup python main.py -v 9 -cid 0 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.2 -b 0.5 -bsm 32 > d6.log 2>&1 &

# nohup python main.py -v 10 -cid 1 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.2 -b 0.8 -bsm 32 > d44.log 2>&1 &
# nohup python main.py -v 11 -cid 1 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.2 -b 0.8 -bsm 32 > d55.log 2>&1 &
# nohup python main.py -v 12 -cid 1 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.2 -b 0.8 -bsm 32 > d66.log 2>&1 &

# nohup python main.py -v 13 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.3 -b 0.5 -bsm 32 > d444.log 2>&1 &
# nohup python main.py -v 14 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.3 -b 0.5 -bsm 32 > d555.log 2>&1 &
# nohup python main.py -v 15 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.3 -b 0.5 -bsm 32 > d666.log 2>&1 &

# nohup python main.py -v 16 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.3 -b 0.8 -bsm 32 > d4444.log 2>&1 &
# nohup python main.py -v 17 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.3 -b 0.8 -bsm 32 > d5555.log 2>&1 &
# nohup python main.py -v 18 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.3 -b 0.8 -bsm 32 > d6666.log 2>&1 &

# nohup python main.py -v 19 -cid 0 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.5 -b 0.5 -bsm 32 > d44444.log 2>&1 &
# nohup python main.py -v 20 -cid 1 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.5 -b 0.5 -bsm 32 > d55555.log 2>&1 &
# nohup python main.py -v 21 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.5 -b 0.5 -bsm 32 > d66666.log 2>&1 &

# nohup python main.py -v 22 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.5 -b 0.8 -bsm 32 > d444444.log 2>&1 &
# nohup python main.py -v 23 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.5 -b 0.8 -bsm 32 > d555555.log 2>&1 &
# nohup python main.py -v 24 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_DER_2000" -ms 2000 -a 0.5 -b 0.8 -bsm 32 > d666666.log 2>&1 &

# --------

nohup python main.py -v 1 -cid 0 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_D_ST" -ms 2000 -a 0.5 -b 0.8 -bsm 32 --ent_weight 0.1 > d10.log 2>&1 &
nohup python main.py -v 1 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_D_ST" -ms 2000 -a 0.5 -b 0.8 -bsm 32 --ent_weight 0.075 > d20.log 2>&1 &
nohup python main.py -v 1 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_D_ST" -ms 2000 -a 0.5 -b 0.8 -bsm 32 --ent_weight 0.05 > d30.log 2>&1 &
nohup python main.py -v 1 -cid 0 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_D_ST" -ms 2000 -a 0.5 -b 0.8 -bsm 32 --ent_weight 0.025 > d40.log 2>&1 &
nohup python main.py -v 1 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_D_ST" -ms 2000 -a 0.5 -b 0.8 -bsm 32 --ent_weight 0.01 > d50.log 2>&1 &
nohup python main.py -v 1 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_D_ST" -ms 2000 -a 0.5 -b 0.8 -bsm 32 --ent_weight 0.0075 > d60.log 2>&1 &
nohup python main.py -v 1 -cid 0 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_D_ST" -ms 2000 -a 0.5 -b 0.8 -bsm 32 --ent_weight 0.005 > d70.log 2>&1 &
nohup python main.py -v 1 -cid 2 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_D_ST" -ms 2000 -a 0.5 -b 0.8 -bsm 32 --ent_weight 0.0025 > d80.log 2>&1 &
nohup python main.py -v 1 -cid 3 -sn "DER" -dn "EuroSAT" -ts 32 -es 32 -tp 50 -lr 0.03 --logdir "./logs/EuroSAT_D_ST" -ms 2000 -a 0.5 -b 0.8 -bsm 32 --ent_weight 0.001 > d90.log 2>&1 &
