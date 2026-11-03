# Run in Vivado Tcl Console: source D:/FPGA/fruit_trigger_package/create_project.tcl
set root [file dirname [file normalize [info script]]]
set target [file join $root build fruit_trigger]
if {[file exists [file join $target fruit_trigger.xpr]]} {
    error "Project already exists. Open build/fruit_trigger/fruit_trigger.xpr instead."
}
if {[llength [get_projects -quiet]]} {close_project}
create_project fruit_trigger $target -part xc7z020clg400-1
set_property target_language Verilog [current_project]
add_files [file join $root rtl ir_trigger.v]
add_files -fileset constrs_1 [file join $root constraints fruit.xdc]
add_files -fileset sim_1 [file join $root sim tb_ir_trigger.v]
set_property top tb_ir_trigger [get_filesets sim_1]
set_property xsim.simulate.runtime {20us} [get_filesets sim_1]
create_bd_design design_1
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
source [file join $root vendor ps_config.tcl]
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_FABRIC_INTERRUPT {0} CONFIG.PCW_EN_CLK1_PORT {0} CONFIG.PCW_EN_CLK2_PORT {0} CONFIG.PCW_EN_CLK3_PORT {0} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100}] $ps7
make_bd_intf_pins_external [get_bd_intf_pins ps7/DDR]
make_bd_intf_pins_external [get_bd_intf_pins ps7/FIXED_IO]
# Keep canonical PS port names.
set_property name DDR [get_bd_intf_ports DDR_0]
set_property name FIXED_IO [get_bd_intf_ports FIXED_IO_0]
set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0]
set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_GPIO_WIDTH {32} CONFIG.C_ALL_INPUTS {1} CONFIG.C_GPIO2_WIDTH {4} CONFIG.C_ALL_OUTPUTS_2 {1} CONFIG.C_DOUT_DEFAULT_2 {0x00000000}] $gpio
create_bd_cell -type module -reference ir_trigger ir_trigger_0
set ir [create_bd_port -dir I ir_in]
connect_bd_net $ir [get_bd_pins ir_trigger_0/ir_in]
connect_bd_net [get_bd_pins axi_gpio_0/gpio_io_i] [get_bd_pins ir_trigger_0/status]
connect_bd_net [get_bd_pins axi_gpio_0/gpio2_io_o] [get_bd_pins ir_trigger_0/control]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins ir_trigger_0/clk]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} [get_bd_intf_pins axi_gpio_0/S_AXI]
# Reuse the reset domain attached by AXI automation.
set reset_net [get_bd_nets -of_objects [get_bd_pins axi_gpio_0/s_axi_aresetn]]
if {[llength $reset_net] != 1} {error "AXI reset not connected by automation"}
connect_bd_net -net $reset_net [get_bd_pins ir_trigger_0/resetn]
assign_bd_address
validate_bd_design
save_bd_design
set bdfile [get_files */design_1.bd]
generate_target all $bdfile
set wrappers [make_wrapper -files $bdfile -top]
add_files -norecurse $wrappers
set_property top design_1_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "PROJECT CREATED. Run simulation, then source build_bitstream.tcl."
