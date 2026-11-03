`timescale 1ns / 1ps

module ir_trigger #(
    parameter integer FILTER_CYCLES = 1000000
)(
    input  wire        clk,
    input  wire        resetn,
    input  wire        ir_in,
    input  wire [31:0] control,
    output wire [31:0] status
);

    // control[0]: enable
    // control[1]: busy
    // control[2]: acknowledge, write 1 then write 0
    // control[3]: blocked input level (0 or 1)
    wire enable;
    wire busy;
    wire ack;
    wire blocked_level;

    assign enable        = control[0];
    assign busy          = control[1];
    assign ack           = control[2];
    assign blocked_level = control[3];

    // Synchronize the asynchronous sensor input.
    (* ASYNC_REG = "TRUE" *) reg ir_meta;
    (* ASYNC_REG = "TRUE" *) reg ir_sync;

    always @(posedge clk) begin
        if (!resetn) begin
            ir_meta <= 1'b0;
            ir_sync <= 1'b0;
        end else begin
            ir_meta <= ir_in;
            ir_sync <= ir_meta;
        end
    end

    // Accept an input level only after it stays unchanged
    // for FILTER_CYCLES consecutive clock cycles.
    reg        candidate;
    reg        filtered;
    reg        filter_valid;
    reg [31:0] filter_count;

    always @(posedge clk) begin
        if (!resetn) begin
            candidate    <= 1'b0;
            filtered     <= 1'b0;
            filter_valid <= 1'b0;
            filter_count <= 32'd0;
        end else begin
            if (ir_sync != candidate) begin
                candidate    <= ir_sync;
                filter_count <= 32'd0;
                filter_valid <= 1'b0;
            end else if (!filter_valid) begin
                if (filter_count >= FILTER_CYCLES - 1) begin
                    filtered     <= candidate;
                    filter_valid <= 1'b1;
                    filter_count <= 32'd0;
                end else begin
                    filter_count <= filter_count + 1'b1;
                end
            end else begin
                filter_count <= 32'd0;
            end
        end
    end

    wire blocked;
    assign blocked = (filtered == blocked_level);

    reg        pending;
    reg        armed;
    reg [31:0] clear_count;

    // A fresh clear interval is required after busy/ack.
    // A pending event stays set until acknowledged.
    always @(posedge clk) begin
        if (!resetn) begin
            pending     <= 1'b0;
            armed       <= 1'b0;
            clear_count <= 32'd0;
        end else if (!enable || ack) begin
            pending     <= 1'b0;
            armed       <= 1'b0;
            clear_count <= 32'd0;
        end else if (busy || pending) begin
            armed       <= 1'b0;
            clear_count <= 32'd0;
        end else if (!armed) begin
            if (filter_valid && !blocked &&
                (ir_sync != blocked_level)) begin
                if (clear_count >= FILTER_CYCLES - 1) begin
                    armed       <= 1'b1;
                    clear_count <= 32'd0;
                end else begin
                    clear_count <= clear_count + 1'b1;
                end
            end else begin
                clear_count <= 32'd0;
            end
        end else begin
            clear_count <= 32'd0;
            if (filter_valid && blocked) begin
                pending <= 1'b1;
                armed   <= 1'b0;
            end
        end
    end

    // status[0]: pending event
    // status[1]: filtered sensor level
    // status[2]: armed
    // status[3]: synchronized sensor level
    // status[4]: filter valid
    // status[5]: blocked (valid only when status[4] = 1)
    assign status = {
        26'd0,
        blocked,
        filter_valid,
        ir_sync,
        armed,
        filtered,
        pending
    };

endmodule
