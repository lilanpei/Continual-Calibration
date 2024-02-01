# nohup python main.py --cuda_id 1 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining > joint.log 2>&1 &
# nohup python main.py --cuda_id 1 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining -stcm > joint_self.log 2>&1 &
# nohup python main.py --cuda_id 1 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining -ppcm > joint_post.log 2>&1 &
# nohup python main.py --cuda_id 2 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Naive > naive.log 2>&1 &
# nohup python main.py --cuda_id 2 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Naive -stcm > naive_self.log 2>&1 &
# nohup python main.py --cuda_id 2 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Naive -ppcm > naive_post.log 2>&1 &
# nohup python main.py --cuda_id 2 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Naive -ppcm -ppdm > naive_post_mixed.log 2>&1 &
# nohup python main.py --cuda_id 3 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 > replay.log 2>&1 &
# nohup python main.py --cuda_id 3 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -stcm > replay_self.log 2>&1 &
# nohup python main.py --cuda_id 3 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -ppcm > replay_post.log 2>&1 &
# nohup python main.py --cuda_id 3 --logdir ./logs/SplitCIFAR100_ne10_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -ppcm -ppdm > replay_post_mixed.log 2>&1 &

nohup python main.py -v 1 --cuda_id 1 --logdir ./logs/F_SplitCIFAR100_3 -ts 256 -es 256 -tp 200 -lr 0.01 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining > joint_1.log 2>&1 &
nohup python main.py -v 2 --cuda_id 1 --logdir ./logs/F_SplitCIFAR100_3 -ts 256 -es 256 -tp 200 -lr 0.01 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining > joint_2.log 2>&1 &
nohup python main.py -v 3 --cuda_id 2 --logdir ./logs/F_SplitCIFAR100_3 -ts 256 -es 256 -tp 200 -lr 0.01 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining > joint_3.log 2>&1 &

nohup python main.py -v 1 --cuda_id 2 --logdir ./logs/F_SplitCIFAR100_3 -ts 256 -es 256 -tp 200 -lr 0.01 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining -stcm --ent_weight 0.025 > joint_st1.log 2>&1 &
nohup python main.py -v 2 --cuda_id 3 --logdir ./logs/F_SplitCIFAR100_3 -ts 256 -es 256 -tp 200 -lr 0.01 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining -stcm --ent_weight 0.025 > joint_st2.log 2>&1 &
nohup python main.py -v 3 --cuda_id 3 --logdir ./logs/F_SplitCIFAR100_3 -ts 256 -es 256 -tp 200 -lr 0.01 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining -stcm --ent_weight 0.025 > joint_st3.log 2>&1 &
