PYNQ-Z2 红外触发工程包（Vivado 2026.1 目标版本）

这是源码和工程生成脚本，不是已综合的 bitstream。本环境没有 Vivado，未执行 Vivado 工程生成、综合或实现；最终需在你的电脑验证。使用现有 PYNQ SD 启动，不重烧 SD，不替换 BOOT.BIN。

1. 解压后目录固定为 D:/FPGA/fruit_trigger_package，避免中文路径。
2. Vivado 底部 Tcl Console 输入：
   source D:/FPGA/fruit_trigger_package/create_project.tcl
   脚本会关闭当前工程但不删除它；先保存现有工程。
3. 出现 PROJECT CREATED 后，工程位于 build/fruit_trigger/fruit_trigger.xpr。
   再次使用直接 Open Project 打开该 xpr，不必重复运行创建脚本。
4. 左侧 Run Simulation -> Run Behavioral Simulation，等待 PASS。
   仿真 FILTER_CYCLES=4 加快测试；实际硬件默认 2,000,000 周期/100MHz=20ms。
5. 关闭仿真，在 Tcl Console 输入：
   source D:/FPGA/fruit_trigger_package/build_bitstream.tcl
6. 成功后 output 内得到同名 fruit_trigger.bit 和 fruit_trigger.hwh。
   将两个文件上传到板子 Jupyter 的同一文件夹。
7. 停止以前按钮程序，在该文件夹运行 python/test_ir.py 的内容。
   无遮挡/遮挡时观察 filtered 值；valid=1 后记录。
8. 修改 trigger_demo.py 的 BLOCKED_LEVEL，与实测遮挡电平一致。
   停止 test_ir 后运行 trigger_demo；它只打印事件，不会拍照。
9. 完整拍照程序已提供 python/capture_ir.py：先修改 BLOCKED_LEVEL。
   将它上传到 .bit/.hwh 同一目录，在 Jupyter 新代码框执行：
   %run capture_ir.py
   必须停止前面的信号测试和 trigger_demo，避免重复控制。
   不要再次运行旧 BaseOverlay/按钮代码，否则会覆盖 PL 设计。
10. 电脑接收程序：可保留之前已修复 /health 的 receiver。
    或将包内 python/receiver.py 复制到 D:/fruit_project/receiver.py。
    在电脑 PowerShell 执行：
    D:/fruit_ai/.venv/Scripts/python.exe -m pip install flask pillow
    D:/fruit_ai/.venv/Scripts/python.exe D:/fruit_project/receiver.py
    保持窗口运行，浏览器打开 http://127.0.0.1:5000/health 检查 ok。
    板子IP保持192.168.2.99，电脑以太网IP保持192.168.2.10。
    电脑照片保存 D:/fruit_project/received。
    首次只做接收，不运行AI，也不动作舵机。
    拍照约预热3秒；整个忙期间需物体留在拍摄位置。
    网络上传失败会保留SD文件，下次启动程序先重试未上传记录。
    摄像头驱动的read调用可能阻塞，3秒是预热目标而非硬超时保证。

接线：兼容 3.3V 的传感器信号白线 -> Arduino A0（Y11）。
传感器地与板子 GND 共地；红线按传感器额定供电连接。
板子 GPIO 不能输入 5V；此工程不做电平转换。
不接 PCA9685/舵机；USB 相机和网线保持原连接。

功能：双触发器同步、20ms 稳定滤波、启动必须先看到无遮挡、遮挡一次锁存、
软件确认清除、忙时忽略新物体、忙结束后要先看到无遮挡才重新允许触发。
单一挂起槽，不排队；连续高速通过的物体可能被忽略，后续需要输送节拍管理。
触发标志不会因物体移开而丢失；事件计数为24位循环计数。

寄存器 axi_gpio_0：
0x00 只读状态：bit0同步原始电平 bit1滤波电平 bit2挂起 bit3已准备 bit4滤波有效
              bit31:8 事件计数。
0x04 输入方向，写 0xFFFFFFFF。
0x08 控制：bit0使能 bit1确认（待挂起清零后撤销）bit2忙 bit3遮挡电平。
0x0C 输出方向，写0。

如果 Tcl 报错，不要跳过错误继续生成；保存完整第一条 ERROR 和上下文。
不要通过 Hardware Manager 重新初始化 PS；这里只在现有 PYNQ Linux 中加载 Overlay。
