# python main.py -sn "JointTraining" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" 
# python main.py -sn "JointTraining" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -stcm
# python main.py -sn "JointTraining" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm
# python main.py -sn "JointTraining" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppvs
# python main.py -sn "JointTraining" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppms
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3"
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -stcm
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppdm
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppvs
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppvs -ppdm
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppms
# python main.py -sn "Naive" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppms -ppdm
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3"
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -stcm
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppdm
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppvs
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppvs -ppdm
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppms
# python main.py -sn "Replay" -cid 1 -tp 20 -ep --logdir "./logs/SplitMNIST_logs/run5_epochs_20_ep_3" -ppcm -ppms -ppdm

# ------------

# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.1 -b 0.5 -bsm 32 > dm4.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.1 -b 0.5 -bsm 32 > dm5.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.1 -b 0.5 -bsm 32 > dm6.log 2>&1 &

# nohup python main.py -v 4 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.1 -b 0.8 -bsm 32 > dm44.log 2>&1 &
# nohup python main.py -v 5 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.1 -b 0.8 -bsm 32 > dm55.log 2>&1 &
# nohup python main.py -v 6 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.1 -b 0.8 -bsm 32 > dm66.log 2>&1 &

# nohup python main.py -v 7 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.2 -b 0.5 -bsm 32 > dm4.log 2>&1 &
# nohup python main.py -v 8 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.2 -b 0.5 -bsm 32 > dm5.log 2>&1 &
# nohup python main.py -v 9 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.2 -b 0.5 -bsm 32 > dm6.log 2>&1 &

# nohup python main.py -v 10 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.2 -b 0.8 -bsm 32 > dm44.log 2>&1 &
# nohup python main.py -v 11 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.2 -b 0.8 -bsm 32 > dm55.log 2>&1 &
# nohup python main.py -v 12 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.2 -b 0.8 -bsm 32 > dm66.log 2>&1 &

# nohup python main.py -v 13 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.3 -b 0.5 -bsm 32 > dm444.log 2>&1 &
# nohup python main.py -v 14 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.3 -b 0.5 -bsm 32 > dm555.log 2>&1 &
# nohup python main.py -v 15 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.3 -b 0.5 -bsm 32 > dm666.log 2>&1 &

# nohup python main.py -v 16 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.3 -b 0.8 -bsm 32 > dm4444.log 2>&1 &
# nohup python main.py -v 17 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.3 -b 0.8 -bsm 32 > dm5555.log 2>&1 &
# nohup python main.py -v 18 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.3 -b 0.8 -bsm 32 > dm6666.log 2>&1 &

# nohup python main.py -v 19 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.5 -b 0.5 -bsm 32 > dm44444.log 2>&1 &
# nohup python main.py -v 20 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.5 -b 0.5 -bsm 32 > dm55555.log 2>&1 &
# nohup python main.py -v 21 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.5 -b 0.5 -bsm 32 > dm66666.log 2>&1 &

# nohup python main.py -v 22 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.5 -b 0.8 -bsm 32 > dm444444.log 2>&1 &
# nohup python main.py -v 23 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.5 -b 0.8 -bsm 32 > dm555555.log 2>&1 &
# nohup python main.py -v 24 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_HP" -ms 2000 -a 0.5 -b 0.8 -bsm 32 > dm666666.log 2>&1 &

# ------------

# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.1 > md1.log 2>&1 &
# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.075 > md2.log 2>&1 &
# nohup python main.py -v 1 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.05 > md3.log 2>&1 &
# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.025 > md4.log 2>&1 &
# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.01 > md5.log 2>&1 &
# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.0075 > md6.log 2>&1 &
# nohup python main.py -v 1 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.005 > md7.log 2>&1 &
# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.0025 > md8.log 2>&1 &

# nohup python main.py -v 2 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.01 > md52.log 2>&1 &
# nohup python main.py -v 3 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.01 > md53.log 2>&1 &

# nohup python main.py -v 4 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.005 > md.log 2>&1 &
# nohup python main.py -v 5 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.005 > md2.log 2>&1 &
# nohup python main.py -v 6 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.005 > md3.log 2>&1 &

# nohup python main.py -v 2 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.0025 > md82.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_ST" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -stcm --ent_weight 0.0025 > md83.log 2>&1 &

# ------------

# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_MS" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppms > mdms_1.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_MS" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppms > mdms_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_MS" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppms > mdms_3.log 2>&1 &

# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_VS" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppvs > mdvs_1.log 2>&1 &
# nohup python main.py -v 2 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_VS" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppvs > mdvs_2.log 2>&1 &
# nohup python main.py -v 3 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_VS" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.01 -mi 100 -ppcm -ppvs > mdvs_3.log 2>&1 &

# nohup python main.py -v 1 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_TS" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.01 -mi 100 -ppcm > mdts_1.log 2>&1 &
# nohup python main.py -v 2 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_TS" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.01 -mi 100 -ppcm > mdts_2.log 2>&1 &
# nohup python main.py -v 3 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_TS" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.01 -mi 100 -ppcm > mdts_3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_MS2" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppms > mdms2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_MS2" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppms > mdms2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_MS2" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppms > mdms2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_VS2" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppvs > mdvs2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_VS2" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppvs > mdvs2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_VS2" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppvs > mdvs2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_TS2" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm > mdts2_1.log 2>&1 &
# nohup python main.py -v 2 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_TS2" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm > mdts2_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_TS2" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm > mdts2_3.log 2>&1 &

# nohup python main.py -v 1 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_MS_MD" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppdm -ppms > mdmsmd_1.log 2>&1 &
# nohup python main.py -v 2 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_MS_MD" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppdm -ppms > mdmsmd_2.log 2>&1 &
# nohup python main.py -v 3 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_MS_MD" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppdm -ppms > mdmsmd_3.log 2>&1 &

# nohup python main.py -v 1 -cid 0 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_VS_MD" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppdm -ppvs > mdvsmd_1.log 2>&1 &
# nohup python main.py -v 2 -cid 1 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_VS_MD" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppdm -ppvs > mdvsmd_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_VS_MD" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppdm -ppvs > mdvsmd_3.log 2>&1 &

# nohup python main.py -v 1 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_TS_MD" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppdm > mdtsmd_1.log 2>&1 &
# nohup python main.py -v 2 -cid 3 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_TS_MD" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppdm > mdtsmd_2.log 2>&1 &
# nohup python main.py -v 3 -cid 2 -sn "DER" -dn "SplitMNIST" -tp 20 --logdir "./logs/SplitMNIST_D_TS_MD" -ms 2000 -a 0.3 -b 0.8 -bsm 32 -lrpp 0.001 -mi 100 -ppcm -ppdm > mdtsmd_3.log 2>&1 &

