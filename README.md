# 雀魂画面识别与牌效分析工具

这是一个 Windows 本地学习/复盘工具。它可以手动框选雀魂画面区域，按固定 UI 区域裁剪自己的手牌、摸牌和宝牌，用本地模板匹配识别牌面，并按牌效优先给出向听数、有效牌和推荐切牌排序。

工具不包含自动点击、自动出牌、账号接入或隐藏式实战外挂功能。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

如果 Python 3.14 安装依赖失败，建议改用 Python 3.12 创建虚拟环境。

## 启动

```powershell
.\.venv\Scripts\python run_app.py
```

## 第一版使用流程

1. 打开雀魂画面，保持窗口大小稳定。
2. 启动工具，点击“框选画面”，拖拽选择整个雀魂游戏画面。
3. 依次点击“框选手牌区”“框选摸牌”“框选宝牌”，只框住实际牌面区域。
4. 如果有副露，点击“框选副露区”，再点“设置牌数”，输入手牌区张数、摸牌张数和副露区可见张数，例如 `10 0 3`。
5. 点击“从截图裁模板”，选择一张清晰截图，并输入底部从左到右的手牌区立牌 + 1 张摸牌。
6. 也可以手动把牌图模板放入 `data/templates/`。模板文件名使用牌名，例如：

```text
1m.png ... 9m.png
1p.png ... 9p.png
1s.png ... 9s.png
east.png south.png west.png north.png white.png green.png red.png
```

7. 点击“开始识别”。识别结果会显示手牌、摸牌、宝牌、副露、置信度和牌效推荐。

## 配置

默认配置在 `data/config.json`。关键字段：

- `game_region`：屏幕上的游戏区域，点击“框选画面”后自动保存。
- `layout.hand_region`：相对游戏区域的底部手牌区。
- `layout.draw_region`：相对游戏区域的摸牌区。
- `layout.dora_region`：相对游戏区域的宝牌指示牌区。
- `recognition.threshold`：模板匹配最低置信度。

如果你的窗口缩放或皮肤不同，先调整这些区域比例。

## 测试

纯牌效测试不需要安装图像依赖：

```powershell
python -m unittest
```
