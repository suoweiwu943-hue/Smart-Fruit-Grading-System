# 原来的A0对射输入
set_property PACKAGE_PIN Y11 [get_ports ir_in]
set_property IOSTANDARD LVCMOS33 [get_ports ir_in]

# Arduino专用SCL引脚
set_property PACKAGE_PIN P16 [get_ports IIC_0_scl_io]
set_property IOSTANDARD LVCMOS33 [get_ports IIC_0_scl_io]

# Arduino专用SDA引脚
set_property PACKAGE_PIN P15 [get_ports IIC_0_sda_io]
set_property IOSTANDARD LVCMOS33 [get_ports IIC_0_sda_io]

set_property MARK_DEBUG true [get_nets IIC_0_scl_i]
set_property MARK_DEBUG true [get_nets IIC_0_scl_o]
set_property MARK_DEBUG true [get_nets IIC_0_scl_t]
set_property MARK_DEBUG true [get_nets IIC_0_sda_i]
set_property MARK_DEBUG true [get_nets IIC_0_sda_o]
set_property MARK_DEBUG true [get_nets IIC_0_sda_t]
create_debug_core u_ila_0 ila
set_property ALL_PROBE_SAME_MU true [get_debug_cores u_ila_0]
set_property ALL_PROBE_SAME_MU_CNT 1 [get_debug_cores u_ila_0]
set_property C_ADV_TRIGGER false [get_debug_cores u_ila_0]
set_property C_DATA_DEPTH 16384 [get_debug_cores u_ila_0]
set_property C_EN_STRG_QUAL false [get_debug_cores u_ila_0]
set_property C_INPUT_PIPE_STAGES 0 [get_debug_cores u_ila_0]
set_property C_TRIGIN_EN false [get_debug_cores u_ila_0]
set_property C_TRIGOUT_EN false [get_debug_cores u_ila_0]
set_property port_width 1 [get_debug_ports u_ila_0/clk]
connect_debug_port u_ila_0/clk [get_nets [list design_1_i/processing_system7_0/inst/FCLK_CLK0]]
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe0]
set_property port_width 1 [get_debug_ports u_ila_0/probe0]
connect_debug_port u_ila_0/probe0 [get_nets [list IIC_0_scl_i]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe1]
set_property port_width 1 [get_debug_ports u_ila_0/probe1]
connect_debug_port u_ila_0/probe1 [get_nets [list IIC_0_scl_t]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe2]
set_property port_width 1 [get_debug_ports u_ila_0/probe2]
connect_debug_port u_ila_0/probe2 [get_nets [list IIC_0_sda_i]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe3]
set_property port_width 1 [get_debug_ports u_ila_0/probe3]
connect_debug_port u_ila_0/probe3 [get_nets [list IIC_0_sda_t]]
set_property C_CLK_INPUT_FREQ_HZ 300000000 [get_debug_cores dbg_hub]
set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]
set_property C_USER_SCAN_CHAIN 1 [get_debug_cores dbg_hub]
connect_debug_port dbg_hub/clk [get_nets u_ila_0_FCLK_CLK0]
