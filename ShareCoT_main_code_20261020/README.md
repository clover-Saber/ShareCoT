# ShareCoT 主实验代码

本包已按当前修订稿对齐训练公式、输入序列化和目标环境。自适应目标现在严格包含式（6）的 1/2 系数，费用文本明确标注 USD；这些更改会影响后续训练。旧版模型、日志和论文数值仍属于原运行，不能直接视为本修订代码重新训练所得的结果。修改前的代码包另有备份。

仅收录当前主表及三次训练统计所需代码：ShareCoT、Direct-FT、修正后的 Single-CoT、无 SHAP 理由版 DHRD、九个传统/深度基线、API 基线，以及主表的 shuffled 对照。未附带任何数据、理由文本、评分、预测、结果、日志或模型权重。

## 文件

- `code/protocol.py`：固定九特征和特征提示词。
- `code/train_llm.py`、`code/loss_weighting.py`：ShareCoT / Direct-FT 训练及自适应权重。
- `code/train_singlecot.py`：最终 instruction-split 单头混合监督；测试时完整生成理由与标签。
- `code/dhrd_model.py`、`code/train_dhrd.py`：DHRD 双头结构与 1:1 全序列监督训练；理由由外部提供，不能替换为 ShareCoT 的 SHAP 理由。
- `code/baselines_suite.py`：Logistic Regression、Random Forest、XGBoost、LogitBoost、Gradient Boosting。
- `code/run_adaboost.py`：主实验 AdaBoost 实现与验证集搜索。
- `code/baselines_text_suite.py`：使用原始值英文序列化的 Vanilla Transformer、TextCNN、BERT；这三个方法应使用本脚本。
- `code/api_suite.py`：当前九特征的 zero-shot / 4-shot 调用、重试、解析与缓存。
- `code/shuffle_rationales.py`：主表的 seed44 循环错配理由对照。
- `code/main_statistics.py`：主表指标、单模型 CI、三次训练均值和样本标准差、配对检验。
- `configs/main_experiment.json`：训练与统计口径说明；不是额外训练调度器。
- `SOURCE_MANIFEST.json`：来源与代码哈希、可移植性修改说明。

## 环境和输入

按论文使用独立 Python 3.10.0 环境安装 `requirements.txt`，Transformers 固定为 5.3.0。scikit-learn 采用支持 Python 3.10 的 1.7.2（原 1.8.0 要求 Python 3.11 及以上）。PyTorch 需要与目标机器 CUDA 驱动兼容的安装版本；7B 训练需要支持 BF16 的 CUDA GPU。本包不附模型，不自动下载模型。

每个训练运行使用单独的 `--root`，其中预先放入：

```text
ROOT/data/train.csv
ROOT/data/val.csv
ROOT/data/test.csv
ROOT/data/rationales.jsonl
```

CSV 必须含 `sample_id`、`shared_trip_authorized` 和配置中列出的九个特征。目标为 0/1，三个 split 的 sample_id 不重叠。正式主实验使用既定的 8000/1000/1000 划分；本包不重新随机划分已有数据。LLM 训练 CSV 必须恰好包含这 11 列。

若 timestamp 或 POI 在 CSV 中已是可读文字，可直接使用。若它们是离散编号，需在同一 `ROOT/data/semantic_dictionary.json` 提供由数据预处理确定的真实“编号→文字”映射；各训练、验证、测试使用相同字典。可复制 `configs/semantic_dictionary.example.json` 的结构填入真实映射。程序不会猜测编号的含义，也不会从测试标签生成字典；映射文件哈希会写入运行配置。传统数值基线的时间字段仍要求可解析的实际日期时间。

外部 JSONL 的每条理由以 `sample_id` 对齐训练样本，含 `reason`、`accepted`，可含一致的 `label`；ShareCoT 与 Single-CoT 使用正式 SHAP 理由，DHRD 必须使用独立的无 SHAP 教师理由。Direct-FT 为绑定相同输入也要求此文件，但不使用理由训练。语料构建和评审代码不在本次主实验包范围内。

下面在包根目录执行。`ROOT`、`QWEN_DIR`、`BERT_DIR` 需替换为自己的实际路径。不同方法/种子应放在独立 ROOT，防止已有结果和新配置混用。

```bash
python code/train_llm.py --root ROOT --method sharecot --model-dir QWEN_DIR --seed 44 --epochs 5 --weighting adaptive
python code/train_llm.py --root ROOT --method direct_ft --model-dir QWEN_DIR --seed 44 --epochs 5 --weighting fixed_3
python code/train_singlecot.py --root ROOT --model-dir QWEN_DIR --seed 44 --epochs 5
python code/train_dhrd.py --root ROOT --method dhrd_style --model-dir QWEN_DIR --seed 44 --epochs 5 --weighting fixed_1
```

Direct-FT 的 `fixed_3` 是原脚本接口要求；其训练实际只优化分类 BCE。7B 的监督和损失归一化按论文执行：每条理由先对有效目标 token 求平均，再对全部记录平均，缺失理由时同时屏蔽理由损失及其自适应正则项。自适应任务项及正则项均包含式（6）的 1/2，两个初始任务系数为 0.5；固定比例控制仍为 1:1、3:1、5:1。各训练自行选择验证 Macro-F1 最高的检查点。

三次训练分别显式指定 seed 43、44、50；表4沿用44号参考运行。每个种子必须使用相同数据、骨干和训练配置。

```bash
python code/baselines_suite.py --root ROOT --method logistic_regression --seed 44 --device cpu
python code/baselines_suite.py --root ROOT --method random_forest --seed 44 --device cpu
python code/baselines_suite.py --root ROOT --method xgboost --seed 44 --device cpu
python code/baselines_suite.py --root ROOT --method logit_boost --seed 44 --device cpu
python code/baselines_suite.py --root ROOT --method gradient_boosting --seed 44 --device cpu
python code/run_adaboost.py --root ROOT --method adaboost --seed 44 --device cpu
python code/baselines_text_suite.py --root ROOT --method transformer --seed 44 --device cpu --bert-model BERT_DIR --max-length 256
python code/baselines_text_suite.py --root ROOT --method textcnn --seed 44 --device cpu --bert-model BERT_DIR --max-length 256
python code/baselines_text_suite.py --root ROOT --method bert --seed 44 --device cpu --bert-model BERT_DIR --max-length 256
```

神经文本基线仍使用同一个本地 BERT tokenizer。保留 CPU 训练命令以匹配正式主实验配置，不把另一种表格网络当成 Vanilla Transformer 或 TextCNN。

## 主表的 shuffled 对照

先把同一正式划分放到独立的 SHUFFLED_ROOT，复制相同训练和验证 CSV。只将训练理由按原实验的循环错排重新配对，保留接收样本的分类标签和理由覆盖掩码：

```bash
python code/shuffle_rationales.py --input MAIN_ROOT/data/rationales.jsonl --output SHUFFLED_ROOT/data/rationales.jsonl
python code/train_llm.py --root SHUFFLED_ROOT --method sharecot --model-dir QWEN_DIR --seed 44 --epochs 5 --weighting adaptive
```

## API 基线

先按 `.env.example` 在终端设置环境变量。本包不会读取服务器 `.env`，不自动建立 SSH 代理，也不会在整理/验证时发起 API 请求。

```bash
python code/api_suite.py --root API_ROOT --workers 6
python code/api_suite.py --root NEW_GEMINI_ROOT --workers 6 --models gemini-3.1-pro-preview --shots 0
```

四样本条件下，API_ROOT 的 train.csv 和 rationales.jsonl 提供训练集内的四个演示（每类两个）；仅运行 zero-shot 时只需 test.csv，以及编号类别需要的语义字典。sample_id 始终按字符串读取，保留前导零。test.csv 决定实际评估样本。本包不附带100条或1000条样本，也不拼接历史测试结果。API别名、随机性和服务端版本可能随提供方变化；若重跑同一条件，使用新ROOT。缓存绑定脚本、序列化协议、语义字典和实际服务 endpoint；任一条件改变均拒绝复用原配置。temperature=0，输出上限512 token。

## 主实验统计

```bash
python code/main_statistics.py model --predictions PREDICTIONS.csv --resamples 1000
python code/main_statistics.py paired --reference SHARECOT.csv --comparator DIRECT_FT.csv --resamples 5000
python code/main_statistics.py paired --reference SHARECOT.csv --comparator SHUFFLED.csv --resamples 5000
python code/main_statistics.py multiseed --predictions SEED43.csv SEED44.csv SEED50.csv
```

预测文件需含 `sample_id,y_true,y_pred`。配对检验会按编号对齐并验证真实标签一致。差值统计量为 ShareCoT Macro-F1 减去对照；CI 为配对 percentile bootstrap，置换为逐记录0.5概率交换两预测，采用双侧加一校正。统计模块不嵌入任何实验值。

这些脚本运行后会在用户指定目录生成结果；交付包本身不包含结果。`SOURCE_MANIFEST.json` 列明全部整理改动。完整训练复现仍需外部正式数据、语料、本地骨干与对应硬件环境。

## 基线协议说明

- Single-CoT 按混合监督训练：有理由记录请求并监督 `reason,pred`；标签记录请求并监督 `pred`；验证和测试请求 `reason,pred`。这仍是单语言模型头，不添加分类头。
- DHRD 使用两层 MLP 分类头、全序列 LM 监督和固定 1:1 联合目标，分类头只读取 prompt 边界；理由须为独立无 SHAP 教师语料。
- TextCNN、Transformer 和 BERT 对相同九项特征使用英文序列化及相同 BERT tokenizer；Qwen 使用中文 chat template。两类序列化均应用类别字典并标明数值单位。论文未规定上述基线细节必须与 ShareCoT 完全相同，因此保留各自算法与训练目标。

## 离线回归检查

```bash
python -B -m unittest discover -s tests -v
```

测试覆盖论文式（6）、缺失理由的损失/正则掩码、系数记录、类别字典、单位、特征输入边界，以及 API 的字符串 ID、zero-shot 独立性和缓存约束。无 GPU 环境下损失公式通过执行实际方法的标量适配器检查，不能代替 PyTorch 自动求导或 Qwen/PEFT 集成测试。API 测试使用离线传输替身，不产生真实调用。
