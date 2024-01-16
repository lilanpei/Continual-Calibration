nohup python main.py --cuda_id 1 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining > joint.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining -stcm > joint_self.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn JointTraining -ppcm > joint_post.log 2>&1 &
nohup python main.py --cuda_id 1 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Naive > naive.log 2>&1 &
nohup python main.py --cuda_id 2 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Naive -stcm > naive_self.log 2>&1 &
nohup python main.py --cuda_id 2 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Naive -ppcm > naive_post.log 2>&1 &
nohup python main.py --cuda_id 2 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Naive -ppcm -ppdm > naive_post_mixed.log 2>&1 &
nohup python main.py --cuda_id 2 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 > replay.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -stcm > replay_self.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -ppcm > replay_post.log 2>&1 &
nohup python main.py --cuda_id 3 --logdir ./logs/SplitCIFAR100_ne5_fixlr -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -ppcm -ppdm > replay_post_mixed.log 2>&1 &