# AI
自己写的人工智能小模型，

- 我的微信：2362651588@qq.com
- 我的 QQ：2362651588
- 我的邮箱：2362651588@qq.com
欢迎提问、交流、互相学习。

python -m pip install --upgrade pip
pip install gym
pip install msgpack
pip install msgpack-python

conda env list
conda create --name universe python=3.5 anaconda
activate universe
conda install pip six libgcc swig
conda install opencv
pip install gym
git clone https://github.com/openai/universe.git
cd ~/universe
pip install -e .


git clone https://github.com/openai/universe.git
cd universe
docker build -t universe .
docker run --privileged --rm -e DOCKER_NET_HOST=172.17.0.1 -v /var/run/docker.sock:/var/run/docker.sock universe pytest
docker run --privileged --rm -it -e DOCKER_NET_HOST=172.17.0.1 -v /var/run/docker.sock:/var/run/docker.sock -v (full path to cloned repo above):/usr/local/universe universe python

可以用的
docker pull openai/universe.flashgames
docker run --privileged --cap-add=SYS_ADMIN --ipc=host -p 5900:5900 -p 15900:15900 quay.io/openai/universe.flashgames
默认的密码是 openai
vnc://localhost:5900