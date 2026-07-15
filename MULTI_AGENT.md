# 多 Agent 协同结构

## 1. 默认拓扑

复杂任务采用“1 个主 Agent + 最多 3 个专项 Agent”的扁平结构：

```text
主 Agent（架构、规划、协调、UI 集成、最终验收）
├── 识别 Agent（recognizer/ 与图像识别测试）
├── 牌效 Agent（mahjong/ 与确定牌例测试）
└── 质量 Agent（测试设计、只读审查、回归与边界检查）
```

简单修复、单文件修改或存在强顺序依赖的任务不启动子 Agent。默认不建立多层 Agent 树；如确有必要继续分派，必须由主 Agent 明确授权并保证总并发数不超过当前环境限制。

## 2. 角色与文件所有权

### 主 Agent：架构、规划、协调与集成

- 先检查 `git status --short`，记录并保护用户已有修改。
- 维护 `app / recognizer / mahjong` 的架构边界，决定模块职责、依赖方向和跨模块接口，避免功能持续堆积到 UI 或单个核心文件。
- 定义需求、非目标、跨模块数据契约和验收条件。
- 根据当前进度、缺陷、测试结果和 `AGENTS.md` 优先级维护后续功能计划，并为每一轮给出范围、依赖、风险和完成标准。
- 独占 `app/main_window.py` 这一集成热点，并负责 `app/screen_select.py`、`run_app.py`、文档的最终修改。
- 汇总子 Agent 结果，重新阅读实际 diff，解决接口差异。
- 运行全量测试，列出真实 Windows、DPI 和截图联调中尚未验证的事项。
- 当前功能验收通过后，主动提出下一项最有价值且依赖已满足的功能；当用户已明确要求持续推进、完成整个阶段或按计划迭代时，主 Agent 可直接拆分下一轮任务并派发给专项 Agent。
- 如果用户只授权了单一功能，主 Agent 只提交下一轮计划和建议，不得未经确认直接扩大实现范围。

### 识别 Agent：图像与识别链路

默认负责：

- `recognizer/capture.py`
- `recognizer/geometry.py`
- `recognizer/models.py`
- `recognizer/config.py`
- `recognizer/recognizer.py`
- `recognizer/templates.py`
- `recognizer/template_builder.py`
- 对应的识别、配置、模板测试
- `data/config.example.json`

职责包括截图、物理坐标、DPI、裁剪、模板匹配、副露结构、置信度、候选结果和诊断图。修改 `recognizer/models.py` 或 `recognizer/config.py` 前，必须先向主 Agent 报告字段、默认值、错误语义和旧配置兼容策略。

### 牌效 Agent：麻将领域算法

默认负责：

- `mahjong/tiles.py`
- `mahjong/shanten.py`
- `mahjong/analyzer.py`
- `tests/test_mahjong.py`

职责包括牌编码、赤五身份、向听数、有效牌、可见牌扣减和切牌排序。不得依赖 PySide6、OpenCV、UI 控件或松散配置字典；每项算法变更必须增加确定牌例。

### 质量 Agent：测试与审查

- 默认只读审查，或只修改任务开始时明确分配给它的 `tests/` 文件。
- 建立验收矩阵，检查兼容性、异常输入、错误降级和回归覆盖。
- 发现生产代码缺陷时提交失败用例、复现步骤和建议接口，不越权修改生产文件。
- 检查项目边界，确保没有自动操作游戏、账号协议接入或规避检测功能。
- 检查是否误提交 `data/config.json`、用户模板、截图或诊断文件。

## 3. 单文件单负责人

所有 Agent 共享同一工作区，修改会立即互相可见，因此：

- 同一时间每个文件只能有一个写入负责人。
- `app/main_window.py` 默认仅由主 Agent 修改。
- `recognizer/recognizer.py` 不允许两个 Agent 同时重构。
- 测试文件也必须在任务开始时明确归属，不因位于 `tests/` 就自动共享。
- 需要改变所有权时，原负责人先停止编辑并报告当前状态，再由主 Agent 重新分配。
- 不覆盖、不还原来源不明的修改；禁止用 `git reset --hard` 或 `git checkout --` 处理冲突。
- 子 Agent 不自行提交、推送或创建 PR，除非用户明确要求且主 Agent 已分派该职责。

## 4. 高冲突契约

以下接口必须先由主 Agent 冻结，再允许并行实现：

- `recognizer/models.py`：识别输出与 UI 的共享契约。
- `recognizer/config.py`：默认值、JSON 兼容、UI 和测试的共享契约。
- `recognizer/recognizer.py`：裁剪、匹配、置信度、降级和诊断的汇合点。
- `mahjong/analyzer.py` 与 `mahjong/shanten.py`：暗牌张数、`open_meld_count` 和向听定义。
- `app/main_window.py`：识别、配置、牌效和展示的最终汇合点。

跨模块新增数据时，先明确 dataclass 名称和字段、默认值、旧调用方兼容方式、错误是抛出还是降级、牌名归一化所在层，以及哪些匹配计入平均置信度。

## 5. 标准协作流程

1. 主 Agent 检查工作区，明确输入、输出、异常、兼容行为和验收条件。
2. 必要时让专项 Agent 先做只读分析，主 Agent 据此冻结唯一接口。
3. 主 Agent 为每个子任务写明目标、非目标、可修改文件、禁止修改文件、交付物和测试命令。
4. 识别、牌效和质量 Agent 在文件范围不重叠时并行工作；主 Agent 同步准备集成。
5. 子 Agent 交付修改文件、接口变化、测试结果、风险及未验证环境。
6. 主 Agent 重新阅读实际改动，独占完成 UI 和跨模块串接。
7. 主 Agent 运行全部测试，检查 `git diff` 与 `git status --short`。
8. 涉及截图、裁剪或 DPI 时，明确列出真实画面验证结果；合成测试不能替代真实联调。
9. 当前功能通过验收后，主 Agent 更新后续计划，选择依赖已满足的最高优先级功能。
10. 主 Agent 仅暂存本轮确认属于当前任务的文件，创建一个独立 Git 提交，并在提交信息中清楚标注本轮完成的功能。
11. 若用户已授权持续迭代，主 Agent 进入下一轮“架构确认 → 任务拆分 → 子 Agent 并行 → 集成验收”；否则只报告计划并等待用户确认。

持续迭代闭环：

```text
架构与优先级评估
        ↓
冻结本轮接口和验收条件
        ↓
派发互不冲突的子任务
        ↓
主 Agent 集成与功能测试
        ↓
回归通过并更新项目计划
        ↓
提交 Git 并标注本轮内容
        ↓
已获持续授权？──是→进入下一轮
        └────────否→报告建议并等待确认
```

建议集成顺序：

```text
数据模型与配置兼容
        ↓
领域算法或识别实现
        ↓
RecognitionResult 等公开输出
        ↓
主窗口集成
        ↓
跨模块回归与真实环境验证
```

## 6. 每轮 Git 提交规范

每完成一轮可独立验收的任务，必须由主 Agent 统一创建一次 Git 提交。子 Agent 不自行提交，避免产生边界不清或包含半成品的提交。

提交前必须：

- 运行本轮定向测试和完整 `unittest`。
- 检查 `git diff` 与 `git status --short`。
- 只暂存本轮任务确认产生的文件，不夹带用户原有修改或其他轮次的工作。
- 确认没有加入 `data/config.json`、用户模板、截图、诊断图或其他本地数据。
- 验证失败、接口尚未集成或修改来源无法确认时，不得提交；应报告阻塞原因。

提交信息使用 `<类型>: <本轮核心内容>` 格式。推荐类型为 `feat`、`fix`、`refactor`、`test`、`docs` 和 `build`，例如：

```text
feat: add structured meld model
fix: isolate meld recognition failures
test: cover open-hand shanten cases
docs: define multi-agent iteration workflow
```

每轮交付报告必须标注：

- 提交哈希和提交信息。
- 提交包含的文件和核心变化。
- 已运行的测试及结果。
- 未纳入提交的现有工作区修改。
- 仍需人工验证或下一轮处理的事项。

## 7. 测试责任

每个专项 Agent 对自己的定向测试负责，主 Agent 对完整回归负责：

- 配置：`.\.venv\Scripts\python -m unittest tests.test_config`
- 模板裁剪：`.\.venv\Scripts\python -m unittest tests.test_template_builder`
- 模板匹配：`.\.venv\Scripts\python -m unittest tests.test_templates`
- 识别结果和错误策略：`.\.venv\Scripts\python -m unittest tests.test_recognizer`
- 牌效：`.\.venv\Scripts\python -m unittest tests.test_mahjong`
- 最终集成：`.\.venv\Scripts\python -m unittest`

修改 UI、计时器、窗口隐藏、DPI 或结果展示时，必须补充人工验证清单。测试报告应区分“自动测试已验证”“代码审查推断”和“仍需真实 Windows/截图验证”。

## 8. 子任务提示模板

```text
角色：<识别 / 牌效 / 质量 Agent>
目标：<可独立验收的结果>
非目标：<明确不做的内容>
允许修改：<完整文件列表>
禁止修改：app/main_window.py 及其他 Agent 所有文件
交付物：<代码 / 测试 / 只读分析>
必须运行：<定向测试命令>
完成后报告：修改文件、公开接口变化、测试结果、风险、未验证事项
不要自行提交或推送 Git；由主 Agent 在本轮验收通过后统一提交。
```

## 9. 自动压缩后的上下文交接

当系统提示即将或已经触发自动上下文压缩、当前对话被摘要，或主 Agent 判断长任务已积累大量上下文时，必须使用 [`context-handoff`](C:/Users/86186/.codex/skills/context-handoff/SKILL.md) 技能完成交接。

交接原则：

- 不因压缩提示立即丢弃正在进行的安全工作；优先把当前最小可验收任务完成、测试并按第 6 节提交 Git。
- 若当前任务无法安全完成，不伪造完成状态；保存已验证进度，明确进行中内容、阻塞点、未提交文件和下一步操作。
- 交接前重新检查实际代码、测试和 Git 状态，不能只依赖压缩摘要或子 Agent 报告。
- 增量更新根目录 `AGENTS.md`，记录架构决策、已完成/进行中事项、验证结果、非显然经验、限制和按优先级排列的下一步。
- 不把 `data/config.json`、用户截图、模板、账号信息或其他忽略的敏感数据写入交接文档或提交。
- 检查 `AGENTS.md` 的 UTF-8 中文显示和实际 diff，避免覆盖用户无关修改。
- 使用可用的任务工具创建新的 Codex 任务，要求新主 Agent 先完整阅读 `AGENTS.md` 和 `MULTI_AGENT.md`、检查 `git status`、保护现有修改，再从进行中事项或最高优先级待办继续。
- 创建新任务成功后，将任务链接或任务指令返回给用户；如果创建工具不可用，输出可直接粘贴到新任务的交接提示。

默认交接提示：

```text
请先完整阅读项目根目录的 AGENTS.md 和 MULTI_AGENT.md，并检查 git status。保留所有现有未提交修改，核对代码和测试的真实状态，然后从 AGENTS.md 记录的进行中工作或最高优先级待办继续。遵循单文件单负责人和逐轮 Git 提交规范；完成后运行相称的测试，并将新的知识、进度和待办增量更新回 AGENTS.md。
```

只有在当前轮任务已完成并提交，或其未完成状态已被准确记录后，当前主 Agent 才结束交接任务。

## 10. 完成标准

- 所有子任务均已完成或明确取消。
- 主 Agent 已检查实际 diff，而不只依赖子 Agent 摘要。
- 跨模块接口一致，没有重复实现或临时兼容分叉。
- 定向测试与全部 `unittest` 通过。
- 主 Agent 已创建只包含本轮内容的 Git 提交，并在交付报告中标注提交哈希、提交信息和文件范围。
- 必需的真实 Windows、125% DPI、1920×1080 和截图检查已完成，或清楚标记为未验证。
- `git status --short` 中没有意外生成的配置、模板、截图或调试文件。
- 最终报告包含修改范围、验证结果、限制和下一步建议。
