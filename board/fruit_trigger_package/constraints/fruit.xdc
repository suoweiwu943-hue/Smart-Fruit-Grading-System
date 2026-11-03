# Arduino A0, verified against Xilinx/PYNQ v3.1 base.xdc.
set_property -dict {PACKAGE_PIN Y11 IOSTANDARD LVCMOS33} [get_ports ir_in]
# External input is asynchronous and passes through a two-flop synchronizer.
set_false_path -from [get_ports ir_in] -to [get_cells -hier -filter {NAME =~ *sync1_reg}]
