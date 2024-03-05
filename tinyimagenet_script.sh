nohup python main.py -v 1 -cid 0 -sn "JointTraining" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_J2" > j1.log 2>&1 &
nohup python main.py -v 2 -cid 1 -sn "JointTraining" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_J2" > j2.log 2>&1 &
nohup python main.py -v 3 -cid 2 -sn "JointTraining" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_J2" > j3.log 2>&1 &

nohup python main.py -v 1 -cid 3 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D2" -ms 500 -a 0.2 -b 0.5 -bsm 32 > d1.log 2>&1 &
nohup python main.py -v 2 -cid 0 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D2" -ms 500 -a 0.2 -b 0.5 -bsm 32 > d2.log 2>&1 &
nohup python main.py -v 3 -cid 1 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D2" -ms 500 -a 0.2 -b 0.5 -bsm 32 > d3.log 2>&1 &

nohup python main.py -v 1 -cid 2 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R2" -ms 500 > r1.log 2>&1 &
nohup python main.py -v 2 -cid 3 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R2" -ms 500 > r2.log 2>&1 &
nohup python main.py -v 3 -cid 0 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R2" -ms 500 > r3.log 2>&1 &

nohup python main.py -v 1 -cid 1 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N2" > n1.log 2>&1 &
nohup python main.py -v 2 -cid 2 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N2" > n2.log 2>&1 &
nohup python main.py -v 3 -cid 3 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N2" > n3.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R_ST" -ms 500 -stcm --ent_weight 0.1 > ris1.log 2>&1 &
# nohup python main.py -v 1 -cid 1 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R_ST" -ms 500 -stcm --ent_weight 0.075 > ris2.log 2>&1 &
# nohup python main.py -v 1 -cid 2 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R_ST" -ms 500 -stcm --ent_weight 0.05 > ris3.log 2>&1 &
# nohup python main.py -v 1 -cid 3 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R_ST" -ms 500 -stcm --ent_weight 0.025 > ris4.log 2>&1 &
# nohup python main.py -v 1 -cid 0 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R_ST" -ms 500 -stcm --ent_weight 0.01 > ris5.log 2>&1 &
# nohup python main.py -v 1 -cid 1 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R_ST" -ms 500 -stcm --ent_weight 0.0075 > ris6.log 2>&1 &
# nohup python main.py -v 1 -cid 2 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R_ST" -ms 500 -stcm --ent_weight 0.005 > ris7.log 2>&1 &
# nohup python main.py -v 1 -cid 3 -sn "Replay" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_R_ST" -ms 500 -stcm --ent_weight 0.0025 > ris8.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N_ST" -stcm --ent_weight 0.1 > nis1.log 2>&1 &
# nohup python main.py -v 1 -cid 1 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N_ST" -stcm --ent_weight 0.075 > nis2.log 2>&1 &
# nohup python main.py -v 1 -cid 2 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N_ST" -stcm --ent_weight 0.05 > nis3.log 2>&1 &
# nohup python main.py -v 1 -cid 3 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N_ST" -stcm --ent_weight 0.025 > nis4.log 2>&1 &
# nohup python main.py -v 1 -cid 0 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N_ST" -stcm --ent_weight 0.01 > nis5.log 2>&1 &
# nohup python main.py -v 1 -cid 1 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N_ST" -stcm --ent_weight 0.0075 > nis6.log 2>&1 &
# nohup python main.py -v 1 -cid 2 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N_ST" -stcm --ent_weight 0.005 > nis7.log 2>&1 &
# nohup python main.py -v 1 -cid 3 -sn "Naive" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_N_ST" -stcm --ent_weight 0.0025 > nis8.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D_ST" -ms 500 -a 0.2 -b 0.5 -bsm 32 -stcm --ent_weight 0.1 > nis1.log 2>&1 &
# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D_ST" -ms 500 -a 0.2 -b 0.5 -bsm 32 -stcm --ent_weight 0.075 > nis2.log 2>&1 &
# nohup python main.py -v 1 -cid 2 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D_ST" -ms 500 -a 0.2 -b 0.5 -bsm 32 -stcm --ent_weight 0.05 > nis3.log 2>&1 &
# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D_ST" -ms 500 -a 0.2 -b 0.5 -bsm 32 -stcm --ent_weight 0.025 > nis4.log 2>&1 &
# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D_ST" -ms 500 -a 0.2 -b 0.5 -bsm 32 -stcm --ent_weight 0.01 > nis5.log 2>&1 &
# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D_ST" -ms 500 -a 0.2 -b 0.5 -bsm 32 -stcm --ent_weight 0.0075 > nis6.log 2>&1 &
# nohup python main.py -v 1 -cid 2 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D_ST" -ms 500 -a 0.2 -b 0.5 -bsm 32 -stcm --ent_weight 0.005 > nis7.log 2>&1 &
# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "TinyImageNet" -ts 32 -es 32 -tp 100 -lr 0.03 --logdir "./logs/TinyImageNet_D_ST" -ms 500 -a 0.2 -b 0.5 -bsm 32 -stcm --ent_weight 0.0025 > nis8.log 2>&1 &