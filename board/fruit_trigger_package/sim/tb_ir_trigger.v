`timescale 1ns/1ps
module tb_ir_trigger;
reg clk=0; always #5 clk=~clk;
reg resetn=0, ir_in=1;
reg [3:0] control=0;
wire [31:0] status;
ir_trigger #(.FILTER_CYCLES(4)) dut(clk,resetn,ir_in,control,status);
task tick; input integer n; integer i; begin for(i=0;i<n;i=i+1) begin @(posedge clk); #1; end end endtask
task check; input condition; input [511:0] message; begin
if (!condition) begin $display("FAIL: %s status=%h",message,status); $fatal(1); end
end endtask
initial begin
 tick(3); resetn=1; control=1; tick(15);
 check(status[3],"arm after clear");
 ir_in=0; tick(1); ir_in=1; tick(12);
 check(!status[2] && status[31:8]==0,"reject short glitch");
 ir_in=0; tick(15); check(status[2] && status[31:8]==1,"first event");
 tick(30); check(status[31:8]==1,"hold gives single event");
 control=7; tick(3); control=5; tick(5);
 check(!status[2],"ack clears pending");
 control=1; tick(15); check(!status[2] && !status[3],"still blocked cannot rearm");
 ir_in=1; tick(15); ir_in=0; tick(15);
 check(status[2] && status[31:8]==2,"release then second event");
 control=7; tick(3); control=5; ir_in=1; tick(15); ir_in=0; tick(15);
 control=1; tick(15); check(!status[2] && status[31:8]==2,"busy event ignored");
 control=0; tick(3); control=1; tick(15);
 check(!status[2] && !status[3],"startup while blocked no event");
 // Test configurable active-high polarity after clear/disable.
 control=0; ir_in=0; tick(15); control=9; tick(5);
 check(status[3],"active high arm"); ir_in=1; tick(15);
 check(status[2] && status[31:8]==3,"active high event");
 $display("PASS: synchronizer/filter/hold/ack/rearm/busy/startup/polarity"); $finish;
end
initial begin #20000; $fatal(1,"timeout"); end
endmodule
