# PYNQ入料触发联调Skill使用示例

## 提交的问题

按`pynq-infeed-trigger-debug`检查本项目入料触发功能，核对Overlay、AXI GPIO通道方向及触发状态，并给出正常入料、持续遮挡、释放后再次入料三种测试步骤。对缺少的信息明确标注，不推测。

## 输入材料

| 材料 | 本次可用信息 |
|---|---|
| 板卡与工程 | 交付包文档指向PYNQ-Z2；有Vivado工程和单独的`fruit_trigger_package`源码包。不同bit/hwh是否同次构建配套，需在实际加载前核对，不能把舵机Overlay默认当作触发工程Overlay。 |
| AXI GPIO | 包内Python示例使用`axi_gpio_0`；通道1配置为输入（方向寄存器偏移`0x04`写`0xFFFFFFFF`），通道2为输出（`0x0C`写0）。 |
| 状态/控制定义 | `0x00`状态：bit0原始同步电平、bit1滤波电平、bit2待处理触发、bit3已准备、bit4输入有效、bit31:8事件计数；`0x08`控制：bit0使能、bit1确认、bit2忙、bit3遮挡有效电平。 |
| 程序与仿真记录 | `board/fruit_trigger_package/python/test_ir.py`、`trigger_demo.py`、`rtl/ir_trigger.v`、`sim/tb_ir_trigger.v`和`VALIDATION.txt`。验证摘要记录于2026-10-27。 |
| 缺少的信息 | 本次没有读取板上当前Overlay的实际文件路径、IP地址映射和实时寄存器值；没有新的上板截图或本次运行日志。 |

## Skill的输出

1. 先确认触发工程的bit/hwh配套并加载，再查看`overlay.ip_dict`确认`axi_gpio_0`名称和地址；不能仅凭IP名称存在推断实物连线正确。
2. 按包内示例配置方向：通道1输入、通道2输出；先关闭触发使能，只读取`gpio.read(0)`，观察通光和遮挡时bit0、bit1、bit4变化。有效遮挡电平按实际读数确认，不能直接假设为低电平。
3. 确认初始化时传感器处于通光状态，bit3就绪后再开启触发。
4. 正常入料：遮挡一次，记录bit2是否置位及bit31:8事件计数是否加1；软件处理后按控制位定义发确认并清除待处理标志。
5. 持续遮挡：保持传感器被遮挡，检查bit2保持为单个待处理事件、计数不重复增加。
6. 释放后再次入料：先通光并确认重新就绪，再遮挡一次，检查产生新事件且计数再加1。
7. 若输入位不随遮挡变化，按供电共地、接线、电平、GPIO地址/方向排查；若输入变化而事件状态不对，再检查PL状态与PS确认/忙状态流程。每一步记录读数和日志。
8. 测试时不要同时运行`test_ir.py`和`trigger_demo.py`等多个写同一GPIO的程序。包内`trigger_demo.py`会自动确认事件，仅用于事件演示；正式拍照流程需使用对应采集程序。

## 操作结果

- **本次Skill调用结果**：按上述输入检查了打包源码和验证摘要，整理出可执行的寄存器检查顺序与三项测试步骤，并指出本次缺少板上实时值和当前Overlay配对信息。
- **已有上板反馈**：项目此前已完成人工联调；用户反馈正常入料、持续遮挡不重复触发、释放后重新入料三种情况通过。没有量化次数和响应时间。本次没有重新操作板卡。
- **已有仿真记录**：`VALIDATION.txt`记录2026-10-27的Icarus Verilog仿真PASS，覆盖启动、滤波、保持、确认、重新使能、忙状态和极性。当前包未附原始控制台输出；本次未重跑。
- **结论边界**：接收服务日志只能说明服务程序运行情况，不能证明Skill曾被调用，也不能替代板端触发验证。本文件记录的是本次对Skill的实际调用与检查；跨项目独立验证仍待完成。

## 建议复跑命令

从交付包根目录，在已安装Icarus Verilog的环境中运行：

```powershell
iverilog -g2012 -s tb_ir_trigger -o sim/tb_ir_trigger.out board/fruit_trigger_package/rtl/ir_trigger.v board/fruit_trigger_package/sim/tb_ir_trigger.v
vvp sim/tb_ir_trigger.out
```

原始输出应保存到报告或日志中。上面的命令是复跑方法，不是本次仿真结果。
