# rl/：StS2 的 PPO 训练栈

这个目录是项目的 Python 强化学习部分。环境就是跑真 `sts2.dll` 的无头引擎，通过 `engine/RL_PROTOCOL.md` 里定义的 JSON-lines 协议对话；引擎负责合法性，每一步给出候选动作列表 `cands`，Python 只回答一个下标。整个栈里没有 LLM，也没有手写的卡牌估值或启发式：观测里的类别值一律哈希成 embedding；奖励包含胜负、楼层推进和（可选的）每个幕 boss；per_floor 和 boss 奖励没有惩罚项。可选的 `shaping=potential/switch` 是势能式塑形，按定义每步会有负值，一整局总和为零，目前的实验都不用。

## 怎么跑

需要一个装好 PyTorch 的 Python 3。一个省事的做法是在仓库根目录建一个继承已有 torch 的 venv：

```bash
python -m venv --system-site-packages .venv
```

所有命令都在仓库根目录下运行：

```bash
python -m pytest -q rl/tests                                   # 单元测试 + 端到端冒烟（CPU，约 10 秒）
python -m rl.train --preset tiny --name smoke                  # CPU 上的小模型 + FakeEngine
CUDA_VISIBLE_DEVICES=2 python -m rl.train --name ironclad_a0 --engine real \
    --engine_dir engine --n_workers 32 --envs_per_worker 8      # 正式训练
python -m rl.train --name ironclad_a0 --resume auto ...                   # 断点续训
python -m rl.evaluate --ckpt runs/ironclad_a0/ckpt/latest.pt --n_seeds 200 --greedy
python -m rl.evaluate --policy random --engine real --n_seeds 200         # 随机基线
python -m rl.bench --device cuda                                          # learner 吞吐基准
```

每个配置项都是命令行参数（见 `config.py`），子配置用点号，例如 `--model.d 384 --caps.max_tok 320`。课程设置是 `--characters Ironclad,Silent --ascension 0`，每局从列表里均匀抽一个角色。运行目录 `runs/<name>/` 里有 `config.json`、`progress.csv`（每次更新一行）、`episodes.csv`（每局一行）和 `ckpt/`；`--tensorboard true` 在装了 tensorboard 时另写一份。

`asc_curriculum` 默认关闭；开启后前沿从 `asc_start`（默认 1）开始，每局以 `asc_frontier_frac`（默认 0.5）的概率打前沿，其余在更低等级均匀抽，前沿最近 `asc_window`（默认 300）个非中止局胜率达到 `asc_threshold`（默认 0.6）就升一级，最高 `asc_max`（默认 10），续训恢复前沿。它可与 `explore_eps`、`lam_build` 同用，不支持战斗分组或决策分叉；开启时 `episodes.csv` 记录 `asc`，`progress.csv` 记录 `asc_frontier` 和各等级的 `w_win_a*`。`rl.eval_watch --ascension N` 固定评测等级并写入 `eval_aN.csv` / `eval_aN/`，只读取 `eval_ckpts/`，归档仍由不带该参数的守护负责。

## 行为指标

`rl.behavior_metrics` 读取一个或多个 `.jsonl.gz` / `.jsonl` 轨迹，写 JSON 并打印简表。比例指标带分子 `k`、样本数 `n`、`rate` 和 95% Wilson `ci95`，均值带 `n`、`total`、`mean`，无样本时为 `null`。不同文件里的同名种子分别计数；决策统计使用全部已记录决策，局均指标和通关率排除中止及未完成的局。

```bash
python -m rl.trace --ckpt runs/base/eval_ckpts/ckpt_000300.pt --engine_dir engine \
    --n_seeds 100 --n_engines 2 --device cpu --behavior_obs --out runs/base/behavior/u300.jsonl.gz
python -m rl.behavior_metrics runs/base/behavior/u300.jsonl.gz --ckpt runs/base/eval_ckpts/ckpt_000300.pt \
    --device cpu --out runs/base/behavior/u300.json
.venv/bin/python tools/behavior_sweep.py --checkpoint base runs/base/eval_ckpts/ckpt_000300.pt \
    --checkpoint grouped runs/grouped/eval_ckpts/ckpt_000300.pt --n_seeds 100 --n_engines 32 --device cuda
```

篝火只统计实际 `rest` 选择，用本次进房首次篝火决策的血量分档：`[0,40%)`、`[40%,70%)`、`[70%,100%)`、满血。地图只统计实际移动，精英可选时包括只有一条路的情况；水晶球只统计工具选择；卡牌奖励跳过率以拿牌或跳过为分母，排除喝药、重抽等中间操作。回血药水的满血使用率以已知血量的回血药水使用为分母，战斗外用药率以全部用药为分母；回血药水 ID 集中在 `behavior_metrics.HEALING_POTIONS`。删牌统计 `deck_remove` 选牌提交的张数，不重复计算商店的入口动作。局末防御牌定义为 `Skill` 且 `v` 中存在 `Block`；旧轨迹缺少变量时该项不计样本，缺少局末牌组时采用最后可见牌组并在 `deck.sources` 标明。

`trace --behavior_obs` 默认关闭，开启后额外保存四类探针决策的完整观测、所用药水及终局完整观测。`behavior_metrics --ckpt` 从检查点的 `cfg.caps` 编码，对至少两个候选的决策用同一策略重算分布；仅从输入轨迹收集同 `act_id`、同幕的 boss，遍历其他 boss，按“观测 × 替换 boss”平均 TV 和首选动作改变率。血量对照用 `max(1, int(hp * 0.7))`，覆盖所有完整观测；缺观测、没有其他 boss 的数量单独报告。批跑脚本依次执行贪心 trace 和含探针的指标计算，默认 100 种子、2 个引擎，把轨迹和结果放在 `runs/<name>/behavior/u<update>.jsonl.gz` / `.json`，汇总 `runs/behavior.csv`（一行一个检查点，含样本数和区间）；`--runs_dir` 和 `--csv` 可改输出位置。

## 接真引擎

引擎一侧只需要满足协议。`EngineProcess` 默认用 `~/.dotnet/dotnet <engine_dir>/src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll` 启动，环境变量 `DOTNET_ROOT=~/.dotnet`，工作目录是 `engine_dir`；要换命令就传 `--engine_cmd '["/path/dotnet","/path/x.dll","--flag"]'`。引擎能跑以后，先用随机策略做一次冒烟：

```bash
python -m rl.engine --engine real --engine_dir engine --episodes 3 --dump tmp/rl_dump
```

它会逐步用 `protocol.check_response` 检查 schema，打印每类决策的延迟、编码耗时、token 数和截断次数，并把每种决策的第一个响应存成 JSON。如果同一进程里重开新局出问题，训练时加 `--restart_every_episode true`。这个开关代价很大：新进程要重新经历 JIT 预热，2026-11-10 在 gamma 上用 16 局 A10 重放量过（`tools/bench_engine_cpu.py`），每局新起进程时引擎每步 16.4 ms CPU，同一进程连打 8 局只要 8.6 ms，逐步行为完全一致。所以加了 `--restart_every_n N`：和 `restart_every_episode` 同用时，每 N 局才换一个新进程（默认 1，即每局都换，和以前一样），评测脚本不指定时沿用检查点里的设置。运行时参数的收益小得多（`DOTNET_TieredPGO=0` 加 64 MB 的 gen0 省约 14%），Debug 和 Release 构建没有差别，crossgen2 预编译有一局的抽牌堆顶信息不同，不能用。`--engine_log true` 会把每个引擎的 stderr 存到 `runs/<name>/engine_logs/`。

## 设计

**数据流。** learner 进程启动 `n_workers` 个 actor 进程，每个 actor 管 `envs_per_worker` 个引擎子进程。actor 对引擎做非阻塞读写（`select` 在所有引擎的 stdout 和 learner 管道上等待），收到响应就当场解析、编码，写进共享内存里属于这个 env 的观测槽（`/dev/shm` 上的 memmap），再给 learner 发一条很小的事件消息。learner 把手头所有已就绪的 env 拼成一批做 GPU 推理，把动作发回去。因为开局和重启都是异步的，某个引擎慢慢 `rl_start` 时不会拖住其他引擎。更新本身是同步的：发满 `rollout_steps` 个动作以后 learner 停止发动作，等这些动作的结果回来，对没结束的 env 用当前观测估值做 bootstrap，然后跑 PPO；这段时间里正在开局的引擎照常工作，所以没有策略滞后。以后要做成完全异步（推理和训练分开），只需要改 `train.Collector`，actor 和协议都不用动。

**出错的处理。** 引擎返回 `type:error`、进程崩溃、超时或者单局步数超过 `max_episode_steps` 时，actor 把最后那个转移标成 `trunc`，然后重启引擎、换新种子开局。`trunc` 的转移不进损失，它之前的转移用它那一步的价值估计做 bootstrap，相当于在这里截断轨迹，既不奖励也不惩罚引擎的 bug。被中止的局在统计里单独计数，不算进胜率。

**观测编码（`encode.py`）。** 每个实体是一个 token，token 是一串"条目"，每个条目是一个哈希后的键加一个可选数值：类别字段写成 `card.id=CARD.BASH`，数值字段写成键 `card.cost` 加值，带数的键值对（卡牌变量、power、意图）写成键 `pw=POWER.STRENGTH_POWER` 加层数。键按字段名加了命名空间，用 blake2b 摘要切出两个独立的 16 位哈希，模型里把哈希表（2^16 行）的两行相加。数值存成 `sign·log1p|x|` 和 `clip(x/scale, ±5)` 两个分量。协议里没见过的字段走同一套通用展开规则，所以新内容、新字段都不会让编码器出错。token 类型有 global、combat、ctx、deck、hand、draw、discard、exhaust、offer_card、picked、relic、potion、player、ally、enemy、orb、map_node、offer_item、DRAW_TOP、ROOM。候选动作也用同样的条目编码，另外带最多 8 个指针：0 号指向主体（手牌、药水、地图节点、商店或奖励里的卡和物品），1 号指向目标敌人，2 号以后指向卡包里的各张卡。候选里内联的卡会变成 offer_card token，和牌组里的卡用同一个卡牌编码器。地图节点带上"是否当前节点""是否合法下一步""和当前行的距离"以及"下游可达的各类节点数"这几项纯结构信息。

**2026-10-28 的观测扩充与两个开关。** 引擎观测按 `docs/02_观测与编码方法论.md` 扩充以后，编码器相应增加了几类条目。对象自身的状态（引擎的 `d` 字段，各个类自己声明的计数、开关、模式）按"所属实体 ID/字段名"做键，因为同名字段在不同类里含义不同；文字值作为类别键。敌人的招式 ID、之后的固定招式、最近三招和本场各招的次数都按"怪物 ID/招式"做键。同一能力有多个实例时，第二个起在键里加 `#k`，层数和变量不会错配。已知顺序的抽牌堆顶牌是单独的 `DRAW_TOP` token，带"第几张"的位置；商店里买不起的商品作为商品 token 进来，带价格和"买不起"标记；水晶球有全局的已揭开物品计数，以及每个候选格的"会揭开几格、周围 5×5 已揭开的是什么"。两个开关只影响编码，旧实验的配置里没有它们，所以行为不变：`caps.map_paths` 给每个地图节点加路径统计（往后每条路上各类房间最多和最少几个、离最近一个几步、一共几条路，纯图结构，和穷举结果在真实地图上逐一核对过）；`caps.pile_counts` 把牌组和三个战斗牌堆当成多重集，相同的牌合成一个 token 带张数，牌组再大也不会碰到上限。开启这些以后单个 token 最多约 35 个条目、单个状态最多约 100 个 token，所以新一轮训练用 `max_tok 192, max_ent 40, max_cent 12`。`tools/obs_sensitivity.py` 检查引擎导出的每个字段都确实影响编码，`tmp/obs_audit/trunc_audit.py` 统计截断。

`caps.enc_rev` 默认 0，保留旧编码；设为 1 时启用能力持牌的实例标签、内联卡去重、遗物候选状态命名和能力层数优先级，并把本幕房间历史编码为 ROOM token。`caps.rooms` 默认 24，保留最近的房间及其房型、ID 和距今顺序；`caps.rooms_in_combat` 默认 False，战斗中不加入这些 token。

**上限（`encode.Caps`）。** 每个观测最多 256 个 token，每个 token 最多 24 个条目，最多 256 个候选（水晶球可以有 2 × 121 个），每个候选最多 8 个条目和 8 个指针。分组上限是：牌组 80、手牌 12、抽牌堆 128、弃牌堆 128、消耗堆 128、候选内联卡 96、多选已选卡 24、遗物 48、药水 8、盟友 4、敌人 8、球 12、地图节点 100（真游戏一幕的地图约 70 个节点，幕初全部可达）。token 按优先级加入（全局、生物、药水、手牌、候选内联物、已选卡、遗物、地图、房间历史、三个牌堆、牌组），所以被候选引用的实体不会被截掉。战斗中默认不放主牌组和地图（`deck_in_combat`、`map_in_combat`），因为战斗中的牌组基本就是三个牌堆加手牌，地图则只影响价值估计；地图只保留从当前节点可达的部分。截断会被计数并打印在训练日志里。上限只影响编码，不影响模型参数，可以随时调整。

**模型（`model.py`）。** token embedding 是类型 embedding 加上三个加权 bag（权重分别是 1、symlog 值、线性值）拼接后的线性投影。三个 bag 合成一个 `[3N × 2^16]` 的稀疏 CSR 矩阵乘哈希表来算，这比 `F.embedding_bag` 的反向传播快约 5 倍，原因是像 `card.cost` 这种几乎每个 token 都有的键会让后者串行化。主干是 pre-LN transformer（默认 d=256、4 层、8 头，没有位置编码，token 是集合）。候选的表示是它自己的 bag 加上被指向 token 的上下文表示（主体、目标、成员各一个投影，成员取均值）和全局 token，再经过一层"候选自注意力 + 对 token 的交叉注意力 + MLP"，最后每个候选出一个 logit，按候选数掩码。价值头读全局 token 和所有 token 的均值。训练和推理都在 bf16 autocast 下进行，残差流、LayerNorm 和损失保持 fp32。

`model.value_tower` 默认 False；开启后价值函数有独立的类型 embedding、token 投影和 Transformer，只与策略共用哈希表，价值损失不经过策略主干。`model.value_layers` 默认 0，表示沿用 `model.layers` 的层数。

`model.mlp` 默认 `gelu`，设为 `swiglu` 后，token、候选和价值塔的 MLP 都改用 SwiGLU；隐藏宽度取原来的 2/3 并向上对齐到 8，使参数量大致相同，logit 头和价值头不变。

**PPO（`ppo.py`）。** GAE 按每个 env 的转移序列单独计算（γ=0.999、λ=0.95），value 用 MAPPO 式的滑动均值方差归一化，优势按批归一化。默认每次更新 32768 个转移、3 个 epoch、minibatch 4096（拆成 1024 的 micro-batch 做梯度累积），clip 0.2，熵系数 0.01，梯度裁剪 1.0，AdamW lr 2.5e-4。`weight_decay` 默认 0，此时优化器只有一个参数组，和以前的实验完全一样；大于 0 时只有主干、候选块和输出头的矩阵权重衰减，哈希表、类型嵌入、LayerNorm 和偏置都不衰减（不常出现的键在不用时也会被衰减掉），从不带衰减的检查点续训（或反过来）时每个参数的 Adam 动量照常接上，每组的衰减以命令行为准。`lr_kl_target` 默认 0（关闭，学习率按 `lr` 和 `lr_final_frac` 的日程走）；大于 0 时每次更新后用每次更新近似 KL 的滑动平均（`lr_kl_ema`）把学习率乘上 (目标 / 平均 KL) 的 `lr_kl_gain` 次方，每次最多变 `lr_kl_step` 倍、限制在 [`lr_kl_min`, `lr_kl_max`] 内，调整后的学习率和 KL 平均存在检查点里、续训接着走。动机是 11-03 的发现：卡住学习的是每次更新的策略变化幅度，而同样的学习率在权重大小不同的网络上对应的幅度差两倍以上。minibatch 按 token 数分桶，减少填充。奖励是胜利 +1、失败 0、`info.total_floor` 每升一层 +0.01，都可配置。`boss_reward` 在每进入新的一幕（也就是打过上一幕 boss）时给分，最后一幕的 boss 就是胜利；`floor_anneal_steps` 让每层奖励在给定的环境步数内线性降到 0（每局开始时决定本局系数），用于从零训练时先靠楼层带路、后期只看 boss 和胜负。续训时这些按步数的日程从检查点的步数接着走。

`teacher` 默认空字符串；指定老师检查点后，每个 PPO micro-batch 附加老师到学生的候选策略 KL 和价值 MSE（Schmitt 等 2018，Kickstarting），采集仍用学生策略，老师的 `caps` 必须完全一致，模型结构可以不同。权重从 `distill_coef`（默认 1）在 `distill_updates`（默认 300）次更新内线性降到 0，之后停止老师前向，续训接着计数；价值项另乘 `distill_value_coef`（默认 1），老师价值先反归一化再按学生当前统计归一化，新实验的 ValueNorm 从老师初始化。

`lam_build` 默认 None，关闭时保留原训练行为；例如 `--lam_build 0.95` 把无 combat token 的构筑决策单独串起来，两次构筑决策之间的战斗合成一个宏步，在这个子序列上做 GAE。实现上先跑原 GAE，再用逐转移 λ 跑一遍（下一个转移是构筑决策时用 `lam_build`，其余用 1），只替换构筑转移的优势和价值目标，战斗转移、截断处理及全批优势归一化不变。该选项不支持同时开启战斗分组或决策分叉；开启时 `progress.csv` 增加 `build_frac`、`adv_std_combat`、`adv_std_build`、`adv_std_build_std`，分别记录构筑占比、合并后两类优势的标准差和原 GAE 的构筑优势标准差，均在优势归一化前统计。

`explore_eps` 默认 0，关闭时保留原训练行为；开启后，每局以 `explore_frac`（默认 0.5）的概率成为探索局，整局标记不变。探索局里，无 combat token 的构筑决策在策略采样后以 `explore_eps` 的概率替换动作，先在当前候选的 `k` 类型间均匀采样，再在该类型内均匀采样，战斗决策不替换。替换后的动作仍按原策略 logits 记录 logp，故意不做重要性采样修正，因此训练有偏差；该选项不支持同时开启战斗分组或决策分叉，评测始终关闭探索。被替换的动作在旧策略下概率可能极小，训练中比值会暴涨，负优势时普通 PPO 不裁剪这一侧，单个样本就能占满梯度；另外，替换出来的动作大多更差，若按普通样本把它们的极小概率继续往下压，网络会整体放大 logit、所有决策的熵一起塌（ε=0.3 的冒烟里 8 次更新熵从 0.60 掉到 0.39，关掉探索的对照稳定在 0.7 左右），所以按自我模仿学习（Oh 等 2018）的做法，被替换且优势为负的样本不进策略损失，只进价值损失。探索还要配 `dual_clip`（例如 3，dual-clip PPO：负优势的目标不低于 `dual_clip × A`）。开启探索时 `approx_kl` 和 `clipfrac` 只统计未被替换的样本，另记 `explored_frac`（被替换样本占比）和 `explored_ratio_mean`（它们的平均比值）。

`shaping_fade_steps` 默认 0（关闭）；大于 0 时，每局开始时按全局环境步数算一个系数，从 `shaping_fade_start` 起在这么多步内从 1 线性降到 `shaping_fade_to`（默认 0），楼层分和 boss 分都乘上它，一局之内不变。动机是高难度上胜负只占回报的一小部分（A10 上胜利 17%、楼层 47%、boss 36%），早期存活的权重压过了后期的强度，而这时胜局已经足够多，可以把目标还原成纯胜负。

`midrun_frac` 默认 0（关闭）；大于 0 时，训练中这个比例的新局不从第一层开始，而是从房间入口存档开局。存档来自这个策略自己在前沿难度上打的完整对局：`midrun_rooms`（默认 `Boss`）里、总楼层不低于 `midrun_min_floor` 的房间入口，每个 worker 用环形缓冲保留最近 `midrun_pool`（默认 128）个，均匀抽取。存档局短，连续读档时共用一个引擎进程，最多 `midrun_reuse`（默认 16）次；完整对局总在新进程里开始。这些局正常进入 PPO，但不计入胜率和课程统计，`episodes.csv` 标 `midrun` 和 `start_floor`，`progress.csv` 记 `midrun_eps`、`midrun_steps`、`midrun_win`。引擎要支持 `rl_save_get`（`engine_v2s`）。不支持同时开战斗分组或决策分叉。

**战斗分组。** `group_k` 默认 1，关闭时保留原采集和训练行为；设为 K>1 时，`group_rooms` 指定的房型（默认 `Elite,Boss`）从 `info.save` 的 room 入口存档各打 K 遍。例如 `--preset tiny --group_k 4 --group_rooms Elite,Boss`。前 K−1 遍胜利后只估战后状态的价值，再用 `rl_load` 读档；该转移用这个价值 bootstrap，GAE 在这里截断，战死则用 0。最后一遍继续整局，中间几遍的死亡不计入局统计，episode 的步数和回报也只保留最后一遍的路径。每遍的结局是战斗内折扣奖励和加战后价值，减去其他遍的平均结局后，与 GAE 按 `group_alpha`（默认 0.5，为 GAE 的权重）混合，再做批内优势归一化；价值目标始终保留原 GAE 回报。`group_alpha=1` 只增加采样。到达 `rollout_steps` 后只继续已采样的分组，不开始新组；缓冲区预留 `max(rollout_steps, 256×环境数)` 步，接近上限时取消剩余读档，把当前遍作为最后一遍，单场超长战斗所需的存储按需扩展。开启时 `progress.csv` 增加 `group_count`、`group_win_rate`（各组内胜率的平均）、`group_outcome_std`（各组结局的总体标准差均值）、`group_extra_frac`（前几遍步数占本次采集步数的比例）。与决策分叉叠加时，支线不分组，主线的精英和 boss 战照常分组；分叉结局只接入每组最后一遍的转移，取消分组或缓冲区余量不足时接入实际继续的当前遍。FakeEngine 同样支持入口存档和确定性回放。

**决策分叉。** `branch_decisions` 默认空字符串，关闭时保持原训练行为；设为 `rest_site` 后，只在 RestSite 进房存档后的首个决策、且不在分组或分叉中时，以 `branch_prob`（默认 1）评估所有 `k=rest` 选项，至少两个才分叉。先按策略采样主线动作，再从同一个入口存档逐条跑其他选项，最后恢复入口执行主线动作并继续整局；主线首步只记录一次，支线强制首步不参与 PPO，其余转移正常训练，支线不写 episode 或胜率、不重启引擎。每条路径到本幕 boss 离开战斗、进入下一幕、死亡或 `branch_max_steps`（默认 600）步为止，非终局只估价值接上；支线复用 `cut/bootstrap` 截断 GAE，主线保持相连。比较期间不再开启新的分叉；支线不开战斗分组，主线照常开启，终点与步数只按最终继续的路径判断，分组前几遍的胜负不结束分叉。若分组最后一遍同时到达分叉终点，先完成分组估值并记录 bootstrap，再估值结束分叉。rollout 边界把进行中的分叉连同主线走到终点，目标在同次更新内算完。结局 Q 是折扣奖励和加终点价值，用 ValueNorm 当前标准差归一化中心化的 Q，再以 `π_old × exp(clip(z,−5,5)/branch_temp)` 得到目标（`branch_temp` 默认 0.1）；仅在篝火选项上归一化策略，附加按分叉数平均的交叉熵，权重 `branch_coef` 默认 1。开启时 `progress.csv` 增加 `branch_count`、`branch_extra_frac`、按选项 key 命名的 `branch_n_*` / `branch_q_*` / `branch_target_*` / `branch_policy_*`（采集时的策略概率）及 `branch_ce`；后出现的选项列会补入表头。评测强制关闭分组与分叉。`--engine fakeproc --fake_args=--rest_test` 提供满血篝火接 boss 的小测试场景：休息后普通打击无法在敌人攻击前杀敌，升级后可以，默认假游戏不受影响。

## 测试和实测数字

`rl/tests` 覆盖了这些内容：FakeEngine 产出所有决策类型且都通过 schema 检查；指针指向正确的 token 类型；未知字段和超大输入不会出错，截断按优先级发生；子进程引擎能跳过日志行，能识别崩溃和挂死；GAE 和逐步的参考实现一致；模型对填充内容和 token 顺序不敏感；训练循环在注入错误和崩溃时能跑完、能续训。

`fake_engine.py` 是一个小而完整的类 StS 游戏（能量、抽牌、格挡、意图、力量和易伤、地图、商店、事件、篝火、宝箱、多选、卡包、水晶球），用来在没有真引擎时验证整个管线。在它上面，均匀随机策略的胜率是 0.3%（300 局），一个只在战斗里贪心出牌的手写策略是 24%（仅用于确认游戏可学，没有进入训练）。默认模型在 b100 上训练 44 次更新（144 万步，约 6 分钟）后，在 300 个留出种子上贪心评测胜率 87%（95% 置信区间 82.7% 到 90.3%），采样评测 82%。

learner 的基准（`rl.bench`，b100 的 GPU 2，d=256、4 层、2186 万参数，状态按中期游戏的规模放大，平均约 45 个 token）：推理在 batch 256 时约 2.4 万样本每秒（含共享内存读取和拷贝到显存约 1.9 万）；一次 256 env × 128 步 × 3 epoch 的 PPO 更新约 7 秒，即 1.3 到 1.4 万样本每秒，显存峰值 8.8 GiB，其中 rollout 观测缓冲占 2.1 GiB。

接真引擎（2026-10-27 部署在 b100 上的 `engine/` 构建，随机初始化的策略）实测如下。同一进程内 `rl_start` 约 0.1 秒，战斗出牌每步中位数 3.5 毫秒，编码一个真实状态中位数 0.47 毫秒；冒烟测试没有发现 schema 问题。256 个引擎（32 个 actor × 8）时，采集阶段每秒约 5 千到 6 千步，learner 的推理循环约 80% 的时间在忙，每次推理平均约 110 个状态；PPO 更新约 5.3 秒，所以算上同步更新，墙钟平均约每秒 3 千步。每个引擎进程常驻内存约 140 MB，256 个共约 35 GB。引擎自身的异常约占 0.1% 的步数（约 6% 的局被中止），每条都带着决策上下文写在 `runs/<name>/engine_errors.log` 里。

## 已知的限制

`keep_ckpts` 只保留最近几个检查点，评测守护如果停摆超过 `keep_ckpts×ckpt_every` 次更新，会漏评中间的检查点。

磁盘写满时训练不退出（2026-11-23，`rl/diskfull.py`）。服务器的数据盘是共用的，经常到 100%。写不进去的检查点跳过，之后每次更新重试一次，盘上已有的检查点和 `latest.pt` 不动（检查点先写临时文件再改名，`latest.pt` 也靠改名切换，不会出现半个文件或者链接丢失）；`progress.csv`、`episodes.csv` 的行留在内存里（最多 20 万行），能写了按原顺序补上，一行要么完整写入要么不写；往日志文件的打印失败就丢掉；评测守护归档失败等下一轮再试，因为磁盘满而失败的评测不计入"三次失败就放弃"。要记住的代价：磁盘一直满着的那段时间没有新检查点，这期间进程如果因为别的原因死掉，会退回到最后一个写成功的检查点；归档也停着，`keep_ckpts×ckpt_every` 次更新以外的检查点就补不回来了。验证脚本是 `tmp/diskfull/e2e.sh`（在一个 700 MB 的磁盘镜像上把盘写到 0 字节再放开），单元测试 `rl/tests/test_diskfull.py`。

吞吐的瓶颈目前在 learner。单个 learner 进程交替做推理和训练：训练的那几秒里 actor 只能等待（正在开局的引擎除外），采集时则是单线程推理循环先饱和，而引擎本身还有好几倍的余量。我试过三种不改架构的办法：`torch.compile` 只快 5% 到 13%；CUDA graph 能正确捕获整个前向（包括稀疏 bag），但在真实规模下前向已经受 GPU 限制，没有收益；按 token 数把一次推理拆成两批，调用次数翻倍，反而更慢。这三样都没有留在代码里。真正的下一步是把推理和训练分开（独立的推理进程或第二块 GPU），再改成 IMPALA 式的异步更新，用 V-trace 一类的离策略修正。地图结构只以节点特征的形式出现，注意力里没有边的信息。被截断的转移直接丢弃，没有用下一个观测做 bootstrap。目前还没有搜索，角色从固定列表中抽取；难度默认固定，也可用 `asc_curriculum` 自动提升。FakeEngine 的数值和真游戏无关，它上面的胜率只说明管线能学，不说明任何关于 StS2 的事情。
