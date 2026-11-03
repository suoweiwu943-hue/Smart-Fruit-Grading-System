# 鲜策智分果蔬分级系统

## 目录结构

- `src/fruit_ai/`：AI训练、推理、采集与数据处理代码；模型权重保留在源码目录。
- `src/fruit_project/`：PC端采集和服务代码。
- `sim/`：仿真说明与运行命令；测试平台位于`board/fruit_trigger_package/sim/tb_ir_trigger.v`，验证摘要见其上一级的`VALIDATION.txt`。该记录对应打包时的工程版本，本次没有重跑。
- `build/bitstreams/`：板端Overlay和调试比特流及匹配硬件描述/探针文件。
- `board/vivado_project/`：Vivado工程，目录内部相对结构保持不变。
- `board/fruit_trigger_package/`：板卡侧配套文件。
- `data/图片整理/素材/`：从fruit_ai、FPGA、fruit_project三个源目录归集并按SHA256去重的图片。
- `data/图片整理/画框/`：归集并去重的YOLO等框标注文件。
- `report/logs/`：必要运行日志。`report/skill使用示例.md`：一次真实的Skill调用记录与操作结果。
- `skill/`：PYNQ入料触发联调Skill。

## 从零构建与运行

### 1.准备源代码

解压交付包，所有命令从解压后的项目根目录执行。原始数据没有改动；项目包内的图片已集中到`data/图片整理/`。AI脚本仍位于`src/fruit_ai/`，PC端脚本位于`src/fruit_project/`。

### 2.准备AI与PC运行环境

使用Python3.10或项目目标板当前支持的Python版本创建独立虚拟环境。按所运行脚本的导入项安装依赖：AI训练/推理使用PyTorch和Ultralytics；PC采集/服务脚本依赖以各脚本import为准（常见包括OpenCV、Flask及PYNQ板端接口）。不要把本机虚拟环境复制进仓库。

从源码目录运行对应脚本，例如：

```powershell
cd src/fruit_ai
python train.py
```

数据预处理或训练脚本会按脚本内配置访问数据路径。若从集中整理目录重新训练，先将`data/图片整理/素材`和`画框`中的配对样本按脚本要求放入YOLO的images/labels目录结构，并检查类别配置和train/val划分。

PC服务端从项目目录启动相应入口：

```powershell
cd src/fruit_project
python receiver_fruit_flow.py
```

如使用其他测试或服务入口，以对应脚本的参数和硬件连接配置为准。

### 3.构建FPGA工程

安装与原工程兼容的AMD Vivado版本，打开：

`board/vivado_project/fruit_trigger.xpr`

等待IP状态就绪后，在Vivado中运行综合、实现和Generate Bitstream。板卡约束位于工程的`fruit_trigger.srcs/constrs_1/new/`下。该工程包含调试逻辑；不要把调试资源报告当作触发模块单独占用。

### 4.下载并运行PYNQ Overlay

已提供的舵机Overlay是`build/bitstreams/fruit_servo.bit`和同目录的`fruit_servo.hwh`。将两个同名文件上传到PYNQ文件系统的同一目录，再在Notebook中执行：

```python
from pynq import Overlay
from pynq.lib.iic import AxiIIC

overlay = Overlay("/home/xilinx/jupyter_notebooks/fruit_trigger/fruit_servo.bit")
overlay.download()
iic = AxiIIC(overlay.ip_dict["axi_iic_0"])
print(hex(overlay.ip_dict["axi_iic_0"]["phys_addr"]))
```

修改路径以实际上传位置为准。调试比特流为`design_1_wrapper_debug.bit`，配套探针文件为`.ltx`；下载调试bit会替换当前Overlay，并使原来的硬件运行状态中断。

### 5.检查运行结果

先检查Notebook是否能实例化Overlay及`axi_iic_0`，再执行单次I2C通信和视觉处理。舵机分拣功能仍需按硬件接线和模块应答结果单独验证。当前记录的日志仅反映已有运行情况，不表示所有功能已完成验证。

## 可复现性说明

本包未包含Python虚拟环境、Vivado缓存和大规模历史运行输出。需在目标机重新安装依赖并重新生成工程产物。训练集/验证集划分和硬件验证次数应以实际实验记录为准。PYNQ入料触发联调Skill初版已随代码交付并附实际使用示例；跨项目验证仍待补充。

## Codex技能包

skill/pynq-infeed-trigger-debug/SKILL.md提供本项目的复现、调试和成果记录指引。
src/fruit_ai/与src/fruit_project/内容保持保留。




