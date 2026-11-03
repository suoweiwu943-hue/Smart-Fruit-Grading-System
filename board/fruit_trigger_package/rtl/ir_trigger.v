`timescale 1ns/1ps
module ir_trigger #(
    parameter integer FILTER_CYCLES = 2000000
)(
    input wire clk,
    input wire resetn,
    input wire ir_in,
    input wire [3:0] control,
    output wire [31:0] status
);
// control: bit0 enable, bit1 ack (hold until pending clears),
// bit2 busy, bit3 blocked_level (0: blocking pulls signal low).
(* ASYNC_REG = "TRUE" *) reg sync1, sync2;
reg filtered, valid;
reg [31:0] filter_count;
reg pending, armed;
reg [23:0] event_count;
wire blocked = filtered == control[3];
assign status = {event_count, 3'b000, valid, armed, pending, filtered, sync2};
always @(posedge clk) begin
    if (!resetn) begin
        sync1 <= 1'b1;
        sync2 <= 1'b1;
    end else begin
        sync1 <= ir_in;
        sync2 <= sync1;
    end
end
// Startup also requires a full stable interval before valid is asserted.
always @(posedge clk) begin
    if (!resetn) begin
        filtered <= 1'b1; valid <= 0; filter_count <= 0;
    end else if (!valid) begin
        if (sync2 != filtered) begin
            filtered <= sync2; filter_count <= 0;
        end else if (filter_count >= FILTER_CYCLES-1) begin
            valid <= 1; filter_count <= 0;
        end else filter_count <= filter_count + 1;
    end else if (sync2 == filtered) begin
        filter_count <= 0;
    end else if (filter_count >= FILTER_CYCLES-1) begin
        filtered <= sync2; filter_count <= 0;
    end else filter_count <= filter_count + 1;
end
always @(posedge clk) begin
    if (!resetn) begin
        pending <= 0; armed <= 0; event_count <= 0;
    end else if (!control[0]) begin
        pending <= 0; armed <= 0;
    end else begin
        if (control[1]) pending <= 0;
        if (!valid || control[2] || pending || control[1]) begin
            armed <= 0;
        end else if (!blocked) begin
            armed <= 1;
        end else if (armed) begin
            pending <= 1; armed <= 0; event_count <= event_count + 1;
        end
    end
end
endmodule
