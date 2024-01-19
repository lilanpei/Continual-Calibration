nohup python main.py --cuda_id 0 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn JointTraining > joint.log 2>&1 &
nohup python main.py --cuda_id 0 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn JointTraining -stcm > joint_self.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn JointTraining -ppcm > joint_post.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Naive > naive.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Naive -stcm > naive_self.log 2>&1 &
nohup python main.py --cuda_id 2 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Naive -ppcm > naive_post.log 2>&1 &
nohup python main.py --cuda_id 2 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Naive -ppcm -ppdm > naive_post_mixed.log 2>&1 &
nohup python main.py --cuda_id 2 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Replay -ms 5000 > replay.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Replay -ms 5000 -stcm > replay_self.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Replay -ms 5000 -ppcm > replay_post.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Replay -ms 5000 -ppcm -ppdm > replay_post_mixed.log 2>&1 &

nohup python main.py --cuda_id 3 --ent_weight 1 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn JointTraining -stcm > joint_1.log 2>&1 &
nohup python main.py --cuda_id 3 --ent_weight 0.1 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn JointTraining -stcm > joint_2.log 2>&1 &
nohup python main.py --cuda_id 3 --ent_weight 0.01 --logdir ./logs/EuroSAT_logs_bs256 -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn JointTraining -stcm > joint_3.log 2>&1 &