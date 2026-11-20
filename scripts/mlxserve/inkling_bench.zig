//! Inkling 2 routed-MoE decode timing: Sushi's EXL3 `moe` against MLX affine gather_qmm (fused gate/up, SwiGLU,
//! down, weighted sum), one layer per call, 48 dependent calls per evaluation with fresh routes per call (the E1
//! method). Hidden 4608, expert width 2304, top-8, 64 resident experts. Runs only with INKLING_BENCH=1.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const io_util = @import("mlx_host").io_util;
const root = @import("root.zig");
const exl3 = @import("expert_exl3.zig");

const H: c_int = 4608;
const I: c_int = 2304;
const E: c_int = 64;
const K: c_int = 8;
const CALLS = 48;
const ROUNDS = 7;

fn on() bool {
    const v = std.c.getenv("INKLING_BENCH") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn free(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

fn op(comptime f: anytype, args: anytype) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(@call(.auto, f, .{&r} ++ args));
    return r;
}

/// Random words for one projection's trellis [E, in/16, out/16, n] (U16) and unit scales.
fn proj(alloc: std.mem.Allocator, rnd: std.Random, in_: c_int, out: c_int, n: c_int) !root.Proj {
    const words = try alloc.alloc(u16, @intCast(E * @divExact(in_, 16) * @divExact(out, 16) * n));
    for (words) |*v| v.* = @truncate(rnd.int(u32));
    const su = try alloc.alloc(u16, @intCast(E * in_));
    const sv = try alloc.alloc(u16, @intCast(E * out));
    for (su) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) 1.0 else -1.0);
    for (sv) |*v| v.* = exl3.f32ToF16Bits(0.01);
    return .{
        .trellis = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ E, @divExact(in_, 16), @divExact(out, 16), n }, 4, .uint16),
        .suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ E, in_ }, 2, .float16),
        .svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ E, out }, 2, .float16),
    };
}

const Affine = struct { wq: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array };

fn affine(s: mlx.mlx_stream, rows: c_int, cols: c_int, bits: c_int) !Affine {
    const dense = try op(mlx.mlx_random_normal, .{ &[_]c_int{ E, rows, cols }, 3, .bfloat16, 0, 0.02, mlx.mlx_array{ .ctx = null }, s });
    defer free(dense);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, dense, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(bits), "affine", .{ .ctx = null }, s));
    var a: Affine = .{ .wq = mlx.mlx_array_new(), .sc = mlx.mlx_array_new(), .bi = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&a.wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&a.sc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&a.bi, triple, 2));
    try mlx.check(mlx.mlx_array_eval(a.wq));
    try mlx.check(mlx.mlx_array_eval(a.sc));
    try mlx.check(mlx.mlx_array_eval(a.bi));
    return a;
}

/// One affine MoE layer as mlx-lm's SwitchGLU runs it: x [1,T,H] bf16, inds [1,T,K], scores [1,T,K] -> [1,T,H].
fn affineLayer(s: mlx.mlx_stream, x: mlx.mlx_array, w13: Affine, w2: Affine, inds: mlx.mlx_array, scores: mlx.mlx_array, T: c_int, bits: c_int) !mlx.mlx_array {
    const no_idx = mlx.mlx_array{ .ctx = null };
    const xe = try op(mlx.mlx_reshape, .{ x, &[_]c_int{ T, 1, 1, H }, 4, s });
    defer free(xe);
    const ix = try op(mlx.mlx_reshape, .{ inds, &[_]c_int{ T, K }, 2, s });
    defer free(ix);
    const gu = try op(mlx.mlx_gather_qmm, .{ xe, w13.wq, w13.sc, w13.bi, no_idx, ix, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(bits), "affine", false, s });
    defer free(gu);
    const g = try op(mlx.mlx_slice, .{ gu, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ T, K, 1, I }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s });
    defer free(g);
    const u = try op(mlx.mlx_slice, .{ gu, &[_]c_int{ 0, 0, 0, I }, 4, &[_]c_int{ T, K, 1, 2 * I }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s });
    defer free(u);
    const sg = try op(mlx.mlx_sigmoid, .{ g, s });
    defer free(sg);
    const silu = try op(mlx.mlx_multiply, .{ g, sg, s });
    defer free(silu);
    const a = try op(mlx.mlx_multiply, .{ silu, u, s });
    defer free(a);
    const d = try op(mlx.mlx_gather_qmm, .{ a, w2.wq, w2.sc, w2.bi, no_idx, ix, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(bits), "affine", false, s });
    defer free(d);
    const w0 = try op(mlx.mlx_reshape, .{ scores, &[_]c_int{ T, K, 1, 1 }, 4, s });
    defer free(w0);
    const w = try op(mlx.mlx_astype, .{ w0, .bfloat16, s });
    defer free(w);
    const dw = try op(mlx.mlx_multiply, .{ d, w, s });
    defer free(dw);
    const sum = try op(mlx.mlx_sum_axis, .{ dw, 1, false, s });
    defer free(sum);
    return op(mlx.mlx_reshape, .{ sum, &[_]c_int{ 1, T, H }, 3, s });
}

const Arm = union(enum) { exl3: root.Bank, affine: struct { w13: Affine, w2: Affine, bits: c_int } };

/// Median per-call microseconds of a 48-call dependent chain (each output * 0 + x feeds the next call).
fn time(s: mlx.mlx_stream, io: std.Io, arm: Arm, x: mlx.mlx_array, zero: mlx.mlx_array, inds: []const mlx.mlx_array, scores: mlx.mlx_array, T: c_int) !u64 {
    var samples: [ROUNDS]u64 = undefined;
    var round: usize = 0;
    while (round < ROUNDS + 2) : (round += 1) {
        var sw = io_util.Stopwatch.init(io);
        var y = try op(mlx.mlx_add, .{ x, zero, s });
        for (inds) |ix| {
            const out = switch (arm) {
                .exl3 => |bank| try root.moe(s, y, bank, ix, scores, exl3.Decode.mul1, false),
                .affine => |af| try affineLayer(s, y, af.w13, af.w2, ix, scores, T, af.bits),
            };
            const z = try op(mlx.mlx_multiply, .{ out, zero, s });
            free(out);
            const next = try op(mlx.mlx_add, .{ z, x, s });
            free(z);
            free(y);
            y = next;
        }
        try mlx.check(mlx.mlx_array_eval(y));
        free(y);
        const ns = sw.read();
        if (round >= 2) samples[round - 2] = ns / CALLS;
    }
    std.mem.sort(u64, &samples, {}, std.sort.asc(u64));
    return samples[ROUNDS / 2] / 1000;
}

test "inkling2 MoE decode: Sushi EXL3 against MLX affine gather_qmm" {
    if (!on()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(20261120);
    const rnd = prng.random();

    const x8f = try op(mlx.mlx_random_normal, .{ &[_]c_int{ 1, 8, H }, 3, .bfloat16, 0, 1, mlx.mlx_array{ .ctx = null }, s });
    defer free(x8f);
    const zero0 = mlx.mlx_array_new_float(0);
    defer free(zero0);
    const zero = try op(mlx.mlx_astype, .{ zero0, .bfloat16, s });
    defer free(zero);

    var exl3_banks: [3]root.Bank = undefined;
    const ns = [_]c_int{ 32, 40, 48 }; // halfwords per tile: K2, K2.5, K3
    for (ns, 0..) |n, i| exl3_banks[i] = .{
        .gate = try proj(alloc, rnd, H, I, n),
        .up = try proj(alloc, rnd, H, I, n),
        .down = try proj(alloc, rnd, I, H, n),
    };
    const bits_list = [_]c_int{ 2, 3, 4 };
    var affines: [3]struct { w13: Affine, w2: Affine } = undefined;
    for (bits_list, 0..) |b, i| affines[i] = .{ .w13 = try affine(s, 2 * I, H, b), .w2 = try affine(s, H, I, b) };

    for ([_]c_int{ 1, 2, 4, 8 }) |T| {
        const x = try op(mlx.mlx_slice, .{ x8f, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, T, H }, 3, &[_]c_int{ 1, 1, 1 }, 3, s });
        defer free(x);
        const sc_h = try alloc.alloc(f32, @intCast(T * K));
        for (sc_h) |*v| v.* = 1.0 / @as(f32, @floatFromInt(K));
        const scores = mlx.mlx_array_new_data(sc_h.ptr, &[_]c_int{ 1, T, K }, 3, .float32);
        defer free(scores);
        var inds: [CALLS]mlx.mlx_array = undefined;
        var distinct_total: usize = 0;
        for (&inds) |*ix| {
            const h = try alloc.alloc(u32, @intCast(T * K));
            var seen: [E]bool = @splat(false);
            var r: usize = 0;
            while (r < T) : (r += 1) { // K distinct experts per row
                var taken: [E]bool = @splat(false);
                var k: usize = 0;
                while (k < K) {
                    const e = rnd.uintLessThan(u32, @intCast(E));
                    if (taken[e]) continue;
                    taken[e] = true;
                    h[r * @as(usize, @intCast(K)) + k] = e;
                    if (!seen[e]) distinct_total += 1;
                    seen[e] = true;
                    k += 1;
                }
            }
            ix.* = mlx.mlx_array_new_data(h.ptr, &[_]c_int{ 1, T, K }, 3, .uint32);
        }
        defer for (inds) |ix| free(ix);
        var line: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&line);
        try w.print("inkling2 T={d} distinct/call={d:.1}:", .{ T, @as(f64, @floatFromInt(distinct_total)) / CALLS });
        for (exl3_banks, 0..) |bank, i| {
            const us = try time(s, t.io, .{ .exl3 = bank }, x, zero, &inds, scores, T);
            try w.print(" exl3_K{s}={d}us", .{ ([_][]const u8{ "2", "2.5", "3" })[i], us });
        }
        for (affines, 0..) |af, i| {
            const us = try time(s, t.io, .{ .affine = .{ .w13 = af.w13, .w2 = af.w2, .bits = bits_list[i] } }, x, zero, &inds, scores, T);
            try w.print(" affine{d}={d}us", .{ bits_list[i], us });
        }
        std.debug.print("{s}\n", .{w.buffered()});
    }
}
