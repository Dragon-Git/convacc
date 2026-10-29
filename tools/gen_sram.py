#!/usr/bin/env python3
"""Generate the SRAM macro and the structural top that ties it to the core.

The core (hw/convacc_mem.sv, from codegen_main with --ram_configurations) speaks
a bare 1RW RAM protocol -- addr/wr_data/wr_mask/rd_mask/we/re plus rd_data, with
no ready/valid -- so the top level here is pure wiring: three `convacc_sram`
instances plus the host image load/readback muxes. There is no datapath, no FSM
and no response registers; the XLS-generated skid buffer owns the RAM handshake.
The SRAM macro is the single replaceable black box.

Both outputs are build artifacts: `make rtl-gen` rewrites them, never edit by
hand.
"""
from __future__ import annotations

import argparse
import os

TEMPLATE = """// GENERATED FILE -- do not edit. Produced by tools/gen_sram.py.
// Parameterised single-write-port / combinational-read SRAM behaviour model.
`default_nettype none

module convacc_sram #(
    parameter INIT_FILE = "",
    parameter DEPTH = {depth},
    parameter WIDTH = {width}
) (
    input  logic                 clk,
    input  logic                 we,
    input  logic [{aw}:0]  waddr,
    input  logic [{ww}:0] wdata,
    input  logic [{sw}:0]  wstrb,
    input  logic                 re,
    input  logic [{aw}:0]  raddr,
    output logic [{ww}:0] rdata
);
    localparam int STRIDE = WIDTH / 8;

    logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end else begin
            for (int i = 0; i < DEPTH; i++) mem[i] = {{WIDTH{{1'b0}}}};
        end
    end

    always_ff @(posedge clk) begin
        if (we) begin
            for (int i = 0; i < STRIDE; i++) begin
                if (wstrb[i]) mem[waddr][i*8 +: 8] <= wdata[i*8 +: 8];
            end
        end
    end

    assign rdata = mem[raddr];
endmodule

`default_nettype wire
"""


def clog2(n: int) -> int:
    return max(1, (n - 1).bit_length())


TOP_TEMPLATE = """// GENERATED FILE -- do not edit. Produced by tools/gen_sram.py.
// Structural top: XLS-generated core (RAM ports) + 3 swappable SRAM macros.
// Pure wiring -- no datapath, no FSM, no response registers.
`default_nettype none

module convacc_top #(
    parameter ACT_INIT = "",
    parameter WGT_INIT = "",
    parameter OUT_INIT = ""
) (
    input  logic         clk,
    input  logic         rst,
    input  logic         start,
    input  logic [13:0]  start_desc,
    output logic         busy,
    output logic         done,
    output logic         error,

    // Host image load / readback (used before and after `start`).
    input  logic         act_we,
    input  logic [13:0]  act_waddr,
    input  logic [63:0]  act_wdata,
    input  logic [7:0]   act_wstrb,
    input  logic         wgt_we,
    input  logic [13:0]  wgt_waddr,
    input  logic [63:0]  wgt_wdata,
    input  logic [7:0]   wgt_wstrb,
    input  logic         out_re,
    input  logic [13:0]  out_raddr,
    output logic [63:0]  out_rdata
);

  // ---------------- core RAM interface ----------------
  logic [31:0] core_act_addr, core_wgt_addr, core_out_addr;
  logic [63:0] core_act_wr_data, core_wgt_wr_data, core_out_wr_data;
  logic [7:0]  core_act_wr_mask, core_wgt_wr_mask, core_out_wr_mask;
  logic [7:0]  core_act_rd_mask, core_wgt_rd_mask, core_out_rd_mask;
  logic        core_act_we, core_wgt_we, core_out_we;
  logic        core_act_re, core_wgt_re, core_out_re;
  logic [63:0] act_rdata, wgt_rdata, out_rdata_i;
  // (registered copies for the core are declared next to the SRAMs below)

  // ---------------- SRAMs ----------------
  logic [13:0] act_sram_raddr, wgt_sram_raddr, out_sram_raddr;
  logic [13:0] act_sram_waddr, wgt_sram_waddr, out_sram_waddr;
  logic [63:0] act_sram_wdata, wgt_sram_wdata, out_sram_wdata;
  logic [7:0]  act_sram_wstrb, wgt_sram_wstrb, out_sram_wstrb;
  logic        act_sram_we,   wgt_sram_we,   out_sram_we;

  // act / wgt: host writes the image, the core only reads.
  assign act_sram_we    = act_we | core_act_we;
  assign act_sram_waddr = core_act_we ? core_act_addr[13:0] : act_waddr;
  assign act_sram_wdata = core_act_we ? core_act_wr_data     : act_wdata;
  assign act_sram_wstrb = core_act_we ? core_act_wr_mask     : act_wstrb;
  assign act_sram_raddr = core_act_addr[13:0];

  assign wgt_sram_we    = wgt_we | core_wgt_we;
  assign wgt_sram_waddr = core_wgt_we ? core_wgt_addr[13:0] : wgt_waddr;
  assign wgt_sram_wdata = core_wgt_we ? core_wgt_wr_data     : wgt_wdata;
  assign wgt_sram_wstrb = core_wgt_we ? core_wgt_wr_mask     : wgt_wstrb;
  assign wgt_sram_raddr = core_wgt_addr[13:0];

  // out: the core writes, the host reads it back after `done`.
  assign out_sram_we    = core_out_we;
  assign out_sram_waddr = core_out_addr[13:0];
  assign out_sram_wdata = core_out_wr_data;
  assign out_sram_wstrb = core_out_wr_mask;
  assign out_sram_raddr = core_out_re ? core_out_addr[13:0] : out_raddr;
  assign out_rdata      = out_rdata_i;

  convacc_sram #(.INIT_FILE(ACT_INIT)) u_act_sram (
      .clk(clk), .we(act_sram_we), .waddr(act_sram_waddr), .wdata(act_sram_wdata),
      .wstrb(act_sram_wstrb), .re(1'b1), .raddr(act_sram_raddr), .rdata(act_rdata));

  convacc_sram #(.INIT_FILE(WGT_INIT)) u_wgt_sram (
      .clk(clk), .we(wgt_sram_we), .waddr(wgt_sram_waddr), .wdata(wgt_sram_wdata),
      .wstrb(wgt_sram_wstrb), .re(1'b1), .raddr(wgt_sram_raddr), .rdata(wgt_rdata));

  convacc_sram #(.INIT_FILE(OUT_INIT)) u_out_sram (
      .clk(clk), .we(out_sram_we), .waddr(out_sram_waddr), .wdata(out_sram_wdata),
      .wstrb(out_sram_wstrb), .re(1'b1), .raddr(out_sram_raddr), .rdata(out_rdata_i));

  // The XLS RAM model is latency 1 -- rd_data must appear one cycle after the
  // request -- while convacc_sram reads combinationally. Registering the read
  // data here gives the core exactly that latency; the host readback below
  // stays combinational.
  logic [63:0] act_rd_data_q, wgt_rd_data_q, out_rd_data_q;
  always_ff @(posedge clk) begin
    act_rd_data_q <= act_rdata;
    wgt_rd_data_q <= wgt_rdata;
    out_rd_data_q <= out_rdata_i;
  end

  // ---------------- generated core ----------------
  logic        core_start_vld, core_start_rdy;
  logic [31:0] core_start_desc;
  logic        core_done, core_done_vld;

  convacc_mem u_core (
      .clk(clk), .rst(rst),
      ._start(core_start_desc), ._start_vld(core_start_vld), ._start_rdy(core_start_rdy),
      ._done(core_done), ._done_vld(core_done_vld), ._done_rdy(1'b1),
      .act_rd_data(act_rd_data_q), .wgt_rd_data(wgt_rd_data_q), .out_rd_data(out_rd_data_q),
      .act_addr(core_act_addr), .act_wr_data(core_act_wr_data), .act_wr_mask(core_act_wr_mask),
      .act_rd_mask(core_act_rd_mask), .act_we(core_act_we), .act_re(core_act_re),
      .wgt_addr(core_wgt_addr), .wgt_wr_data(core_wgt_wr_data), .wgt_wr_mask(core_wgt_wr_mask),
      .wgt_rd_mask(core_wgt_rd_mask), .wgt_we(core_wgt_we), .wgt_re(core_wgt_re),
      .out_addr(core_out_addr), .out_wr_data(core_out_wr_data), .out_wr_mask(core_out_wr_mask),
      .out_rd_mask(core_out_rd_mask), .out_we(core_out_we), .out_re(core_out_re));

  // ---------------- host start/done ----------------
  logic start_pending, done_q;
  always_ff @(posedge clk) begin
    if (rst) begin
      start_pending <= 1'b0;
      done_q        <= 1'b0;
      core_start_desc <= '0;
    end else begin
      if (start) begin
        start_pending <= 1'b1;
        core_start_desc <= {18'b0, start_desc};
        done_q <= 1'b0;
      end else if (core_start_vld && core_start_rdy) begin
        start_pending <= 1'b0;
      end
      if (core_done_vld) done_q <= 1'b1;
    end
  end

  assign core_start_vld = start_pending;
  assign busy  = start_pending;
  assign done  = done_q;
  assign error = 1'b0;

endmodule

`default_nettype wire
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=16384)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.width % 8:
        raise SystemExit("width must be a multiple of 8")
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hw", "convacc_sram.sv"
    )
    text = TEMPLATE.format(
        depth=args.depth,
        width=args.width,
        aw=clog2(args.depth) - 1,
        ww=args.width - 1,
        sw=args.width // 8 - 1,
    )
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out} (depth={args.depth}, width={args.width})")

    top = os.path.join(os.path.dirname(out), "convacc_top.sv")
    with open(top, "w") as f:
        f.write(TOP_TEMPLATE)
    print(f"wrote {top} (structural top: core + {args.depth}-word SRAM macros)")


if __name__ == "__main__":
    main()
