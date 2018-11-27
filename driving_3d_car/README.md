运行:sudo python play.py --num-workers 2 --log-dir neonrace
查看tensorboard输出：http://localhost:6006
http://localhost:15900/viewer/?password=openai
查看进程的输出日志：tail -f neonrace/*.out
查看进程号：cat neonrace/kill.sh
结束进程:source neonrace/kill.sh


open vnc://localhost:5900
password:openai
open vnc://localhost:5901
password:openai