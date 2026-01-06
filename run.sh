nohup /home/scxsz1/.conda/envs/MARL/bin/python -u main.py > running2.log 2>&1 &

# 查看后台进程是否还在
# ps -ef | grep python | grep 你的代码.py
# 实时查看日志（类似终端打印）
# tail -f 运行日志.log

# kill -9 进程ID  # 进程ID从ps命令结果中获取``