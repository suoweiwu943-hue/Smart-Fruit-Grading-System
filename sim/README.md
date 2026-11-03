交付包提供了触发模块仿真测试文件，位置为board/fruit_trigger_package/sim/tb_ir_trigger.v；验证摘要见同目录上一级的VALIDATION.txt。该记录对应打包时的工程版本，本次没有重新运行仿真。

Icarus Verilog参考命令（从交付包根目录执行）：

~~~powershell
iverilog -g2012 -s tb_ir_trigger -o sim/tb_ir_trigger.out board/fruit_trigger_package/rtl/ir_trigger.v board/fruit_trigger_package/sim/tb_ir_trigger.v
vvp sim/tb_ir_trigger.out
~~~

VALIDATION.txt只保存了PASS摘要，没有附原始仿真控制台输出；以上命令供后续复跑，不代表本次执行。Vivado工程生成、综合实现、时序和板级集成未由该验证记录覆盖。
