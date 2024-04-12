# README

## Quick Start

### Download

```shell
git clone --recurse-submodules git@github.com:WoodOxen/Carla-Challenge.git
```

### Environment Configuration
```shell
cd Carla-Challenge
conda create --name carla-challenge python==3.8
pip install -r requirements.txt

# install carla
wget https://leaderboard-public-contents.s3.us-west-2.amazonaws.com/CARLA_Leaderboard_2.0.tar.xz
mkdir ./carla
tar -xf CARLA_Leaderboard_2.0.tar.xz -C ./carla

# remember to link dataset to ./data
```

TODO: dataset

## 文件夹构成

- `CARLA` 、`leaderboard` 和 `scenario_runner` 里是比赛环境，任何更改在提交的时候都会被覆盖。
- `teamcode` 是实现我们的 agent 的代码区
- `workspace_samples` 包含了一些以往参考代码，目前有 `Interfuser`, `TCP`, `LAV`
- `workspace_submit` 是提交代码的工作区，运行 `submit.sh` 可以生成用来提交的 docker image

## Agent 编写和数据采集

参考 `leaderboard/autoagents/human_agent.py` 或者 `workspace_samples/InterFuser/leaderboard/team_code/interfuser_agent.py`. 主要需要实现

- `sensor()`: 传感器配置
- `run_step()`: 接收传感器数据并给出控制信号

还有 `get_entry_point()`, `setup()`, `tick()` 什么的，详情还得看[这里](https://leaderboard.carla.org/get_started/#3-creating-your-own-autonomous-agent)

官方说 `leaderboard 2.0` 比之前多了很多复杂的路况，用规则的agent去提取训练数据很困难，所以给了满分 agent 在这些路况下的行驶数据。可以用 CARLA 的回放功能和这些行驶数据来提取训练数据。提取脚本和行驶数据在 `teamcode/data` 里。

行驶数据的[下载连接](https://leaderboard-logs.s3.us-west-2.amazonaws.com/Scenario+Logs.zip)，或者

    curl -O https://leaderboard-logs.s3.us-west-2.amazonaws.com/Scenario+Logs.zip

下载后解压到 `teamcode/data/Scenario_logs/` 下。要对这部分行驶数据进行提取，在`/teamcode/data`目录下运行`run_extractor.sh`（记得先运行Carla）。提取的数据会生成在`/teamcode/data/database`下。通过`/teamcode/data/extractor_config.py`可以修改数据提取的相关参数，包括待提取的行车记录的log文件路径、生成数据的目标位置、传感器配置、天气和采集帧率等。


下面的文件夹里都有个写好的 `.gitignore`，里面的东西默认不会被上传到仓库上：

- `CARLA/`
- `teamcode/data/Scenario_logs/`
- `teamcode/data/database/`

还有各种很大的压缩文件和权重也写在了主目录的 `.gitignore` 里，默认不会被同步。

## 代码提交

在 `workspace_submit` 里运行 `submit.sh` 生成用来提交的 docker image 后在 [这里](https://eval.ai/web/challenges/challenge-page/2098/submission) 按步骤提交。在 `leaderboard/scripts/Dockerfile.master` 里更改要提交的 agent 路径和环境配置。 

## 快速访问

- [CARLA Challenge 官网](https://leaderboard.carla.org/) | [CARLA 0.9.14 文档](https://carla.readthedocs.io/en/0.9.14/)
- [InterFuser 上届权重下载](http://43.159.60.142/s/p2CN)
- [围观复现 TCP](https://github.com/Kin-Zhang/carla-expert/discussions/2) | [讨论](https://github.com/Kin-Zhang/carla-expert/discussions)
- [TCP官方知乎解读](https://zhuanlan.zhihu.com/p/532665469)
