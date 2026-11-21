//! Inkling 2 routed experts on real kortexa layers and recorded routes: Sushi's rate-grouped decode and prefill on layer 0
//! (all K3) and layer 30 (mixed K2/K3), replaying the picks and MoE inputs `tf-inkling-run taps` recorded. Per layer:
//! decode time by window width, window-against-single bit comparison, prefill time by chunk size, partition bit
//! comparison, and the per-projection outputs the precision check compares against FP64. One layer is resident at a
//! time. Hidden 4608, expert width 2304, top-8, 384 experts. Runs only with INKLING_REAL=1.
//!
//! Environment:
//!   INKLING_KORTEXA  the kortexa pack (checkpoint.json, layers/NN/experts-*.safetensors)
//!   INKLING_TAPS     the taps dir (seq-NNNNN.{tokens.json,eids.bin,xm.bin})
//!   INKLING_OUT      output dir; also holds ref-l{L}.act.bf16 from `ink2q.tfexperts ref`
//!   INKLING_SEQ      the precision sequence (default 0)
//!   INKLING_ROW0     the first of the 64 precision rows in that sequence
//!   INKLING_LEGS     the legs to run, comma separated (default all): `time` (decode and prefill timings), `bits`
//!                    (window and partition bit comparisons), `prec` (the precision outputs)
//!
//! Output: JSON lines on stderr (`decode`, `prefill`, `decode_bits`, `prefill_bits`, plus `legs`, `device`, `loaded` and
//! `precision` lines), and INKLING_OUT/sushi-l{L}.{g.f32,u.f32,y.f16} for the precision rows.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const log = @import("mlx_host").log;
const io_util = @import("mlx_host").io_util;
const root = @import("root.zig");

const kernels = root.kernels;
const mul1 = root.format.Decode.mul1;
const max_groups = root.group_layout.max_groups;

const H: usize = 4608;
const I: usize = 2304;
const E: usize = 384;
const K: usize = 8;
/// Layers in the taps' `eids.bin` rows.
const MODEL_LAYERS: usize = 48;
const CALLS = 48;
const ROUNDS = 7;
const WARMUPS = 2;
const PREFILL_ROUNDS = 5;
const PRECISION_ROWS = 64;
const WIDTHS = [_]usize{ 1, 2, 4, 8, 16 };
const BIT_WIDTHS = [_]usize{ 2, 4, 8, 16 };
const BIT_WINDOWS = 8;
const CHUNKS = [_]usize{ 512, 2048, 4096 };
/// The layers the taps carry MoE inputs for, with each one's index on `xm.bin`'s layer axis.
const REAL_LAYERS = [_]struct { id: usize, xm: usize }{ .{ .id = 0, .xm = 0 }, .{ .id = 30, .xm = 1 } };
/// Router scores of every call: 1/8 per pick.
const eighth: [CHUNKS[CHUNKS.len - 1] * K]f32 = @splat(0.125);

fn env(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

fn on() bool {
    return std.mem.eql(u8, env("INKLING_REAL") orelse return false, "1");
}

fn required(name: [*:0]const u8) ![]const u8 {
    return env(name) orelse {
        std.debug.print("{s} is not set\n", .{name});
        return error.MissingEnv;
    };
}

fn number(name: [*:0]const u8, default: ?usize) !usize {
    if (env(name)) |text| return std.fmt.parseInt(usize, text, 10);
    if (default) |d| return d;
    std.debug.print("{s} is not set\n", .{name});
    return error.MissingEnv;
}

/// The legs of a run: timings need a quiet GPU, the bit comparisons and the precision outputs do not.
const Legs = struct {
    time: bool = false,
    bits: bool = false,
    prec: bool = false,

    fn fromEnv() !Legs {
        const text = env("INKLING_LEGS") orelse return .{ .time = true, .bits = true, .prec = true };
        var legs: Legs = .{};
        var it = std.mem.tokenizeScalar(u8, text, ',');
        while (it.next()) |leg| {
            if (std.mem.eql(u8, leg, "time")) {
                legs.time = true;
            } else if (std.mem.eql(u8, leg, "bits")) {
                legs.bits = true;
            } else if (std.mem.eql(u8, leg, "prec")) {
                legs.prec = true;
            } else {
                std.debug.print("INKLING_LEGS: unknown leg \"{s}\" (time, bits, prec)\n", .{leg});
                return error.BadLegs;
            }
        }
        if (!legs.time and !legs.bits and !legs.prec) return error.BadLegs;
        return legs;
    }
};

const Config = struct { kortexa: []const u8, taps: []const u8, out: []const u8, seq: usize, row0: usize, legs: Legs };

fn config() !Config {
    const legs = try Legs.fromEnv();
    return .{
        .kortexa = try required("INKLING_KORTEXA"),
        .taps = try required("INKLING_TAPS"),
        .out = try required("INKLING_OUT"),
        .seq = try number("INKLING_SEQ", 0),
        // Only the precision leg reads a row range.
        .row0 = try number("INKLING_ROW0", if (legs.prec) null else 0),
        .legs = legs,
    };
}

fn free(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

fn freeAll(arrs: []mlx.mlx_array) void {
    for (arrs) |*a| {
        if (a.ctx != null) free(a.*);
        a.* = .{ .ctx = null };
    }
}

fn op(comptime f: anytype, args: anytype) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    errdefer free(r);
    try mlx.check(@call(.auto, f, .{&r} ++ args));
    return r;
}

/// A new array holding a copy of `data` in `dims` (at most four).
fn upload(comptime T: type, data: []const T, dims: []const usize, dtype: mlx.mlx_dtype) mlx.mlx_array {
    var shape: [4]c_int = undefined;
    for (dims, 0..) |d, i| shape[i] = @intCast(d);
    return mlx.mlx_array_new_data(data.ptr, &shape, @intCast(dims.len), dtype);
}

fn evalAll(arrs: []const mlx.mlx_array) !void {
    const v = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
}

/// `a`'s bytes, evaluated and made contiguous; `held` owns them until the caller frees it.
fn bytesOf(s: mlx.mlx_stream, a: mlx.mlx_array, held: *mlx.mlx_array) ![]const u8 {
    try mlx.check(mlx.mlx_contiguous(held, a, false, s));
    try mlx.check(mlx.mlx_array_eval(held.*));
    const p = mlx.mlx_array_data_uint8(held.*) orelse return error.Unreadable;
    return p[0 .. mlx.mlx_array_size(held.*) * mlx.mlx_array_itemsize(held.*)];
}

fn peakGiB() f64 {
    var v: usize = 0;
    _ = mlx.mlx_get_peak_memory(&v);
    return @as(f64, @floatFromInt(v)) / (1 << 30);
}

fn seconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn median(samples: []u64) u64 {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    return samples[samples.len / 2];
}

// ---- The layer: kortexa experts as rate-grouped banks ----

// Indices into one expert's eight pieces.
const gate_t = 0;
const up_t = 1;
const down_t = 2;
const suh13 = 3;
const svh_g = 4;
const svh_u = 5;
const suh2 = 6;
const svh2 = 7;

const Loaded = struct {
    rb: root.Ragged,
    /// Ragged slot of each expert id: `base[group] + index inside the group`.
    slot_of: [E]u32,

    fn deinit(self: *Loaded) void {
        self.rb.deinit();
        _ = mlx.mlx_clear_cache();
    }
};

fn tensor(map: mlx.mlx_map_string_to_array, idx: usize, comptime name: []const u8) !mlx.mlx_array {
    var buf: [96]u8 = undefined;
    const key = try std.fmt.bufPrintSentinel(&buf, "experts.{d}." ++ name, .{idx}, 0);
    var out = mlx.mlx_array_new();
    if (mlx.mlx_map_string_to_array_get(&out, map, key.ptr) != 0) {
        free(out);
        std.debug.print("kortexa tensor {s} is missing\n", .{key});
        return error.TensorMissing;
    }
    return out;
}

fn hasTensor(map: mlx.mlx_map_string_to_array, key: [*:0]const u8) bool {
    var out = mlx.mlx_array_new();
    defer free(out);
    return mlx.mlx_map_string_to_array_get(&out, map, key) == 0;
}

fn expectArray(a: mlx.mlx_array, dtype: mlx.mlx_dtype, dims: []const c_int) !void {
    if (mlx.mlx_array_dtype(a) != dtype or !std.mem.eql(c_int, mlx.getShape(a), dims)) return error.UnexpectedTensor;
}

/// A trellis [in/16, out/16, n] of int16 words: its n.
fn trellisN(a: mlx.mlx_array, lead: [2]c_int) !c_int {
    const sh = mlx.getShape(a);
    if (mlx.mlx_array_dtype(a) != .int16 or sh.len != 3 or sh[0] != lead[0] or sh[1] != lead[1] or @mod(sh[2], 16) != 0 or sh[2] < 16 or sh[2] > 128)
        return error.UnexpectedTensor;
    return sh[2];
}

fn slice3(s: mlx.mlx_stream, a: mlx.mlx_array, lo: [3]c_int, hi: [3]c_int) !mlx.mlx_array {
    return op(mlx.mlx_slice, .{ a, &lo, 3, &hi, 3, &[_]c_int{ 1, 1, 1 }, 3, s });
}

fn slice1(s: mlx.mlx_stream, a: mlx.mlx_array, lo: c_int, hi: c_int) !mlx.mlx_array {
    return op(mlx.mlx_slice, .{ a, &[_]c_int{lo}, 1, &[_]c_int{hi}, 1, &[_]c_int{1}, 1, s });
}

/// One expert's pieces in the layout Sushi's banks take, as lazy views of the file's tensors: w13's trellis split into
/// gate (output tiles 0..144) and up (144..288), its svh into [0, 2304) and [2304, 4608), suh shared; w2 whole.
/// Returns the expert's bits (n / 16).
fn expertPieces(cpu: mlx.mlx_stream, map: mlx.mlx_map_string_to_array, idx: usize, out: *[8]mlx.mlx_array) !u8 {
    errdefer freeAll(out);
    const w13 = try tensor(map, idx, "w13.trellis");
    defer free(w13);
    const w2 = try tensor(map, idx, "w2.trellis");
    defer free(w2);
    const svh13 = try tensor(map, idx, "w13.svh");
    defer free(svh13);
    out[suh13] = try tensor(map, idx, "w13.suh");
    out[suh2] = try tensor(map, idx, "w2.suh");
    out[svh2] = try tensor(map, idx, "w2.svh");
    const n = try trellisN(w13, .{ @intCast(H / 16), @intCast(2 * I / 16) });
    if (n != try trellisN(w2, .{ @intCast(I / 16), @intCast(H / 16) })) return error.MixedRatePerExpert;
    try expectArray(out[suh13], .float16, &.{@intCast(H)});
    try expectArray(svh13, .float16, &.{@intCast(2 * I)});
    try expectArray(out[suh2], .float16, &.{@intCast(I)});
    try expectArray(out[svh2], .float16, &.{@intCast(H)});

    const words = try op(mlx.mlx_view, .{ w13, .uint16, cpu });
    defer free(words);
    const tiles_in: c_int = @intCast(H / 16);
    const tiles_out: c_int = @intCast(I / 16);
    out[gate_t] = try slice3(cpu, words, .{ 0, 0, 0 }, .{ tiles_in, tiles_out, n });
    out[up_t] = try slice3(cpu, words, .{ 0, tiles_out, 0 }, .{ tiles_in, 2 * tiles_out, n });
    out[down_t] = try op(mlx.mlx_view, .{ w2, .uint16, cpu });
    out[svh_g] = try slice1(cpu, svh13, 0, @intCast(I));
    out[svh_u] = try slice1(cpu, svh13, @intCast(I), @intCast(2 * I));
    return @intCast(@divExact(n, 16));
}

fn stack(s: mlx.mlx_stream, parts: []const mlx.mlx_array) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    return op(mlx.mlx_stack_axis, .{ vec, 0, s });
}

fn freeBank(b: *root.Bank) void {
    for ([_]*root.Proj{ &b.gate, &b.up, &b.down }) |p| {
        free(p.trellis);
        free(p.suh);
        free(p.svh);
    }
}

/// The group's experts stacked into a `Bank`, evaluated; the members' lazy pieces are freed once their bytes are in the
/// stacks, so a layer never holds the file data and a second copy for longer than one group.
fn buildBank(a: std.mem.Allocator, s: mlx.mlx_stream, held: *[E][8]mlx.mlx_array, members: []const u32) !root.Bank {
    const parts = try a.alloc(mlx.mlx_array, members.len);
    defer a.free(parts);
    var st: [8]mlx.mlx_array = @splat(.{ .ctx = null });
    errdefer freeAll(&st);
    for (0..8) |p| {
        for (members, parts) |e, *dst| dst.* = held[e][p];
        st[p] = try stack(s, parts);
    }
    try evalAll(&st);
    for (members) |e| freeAll(&held[e]);
    var up_suh = mlx.mlx_array_new();
    errdefer free(up_suh);
    try mlx.check(mlx.mlx_array_set(&up_suh, st[suh13]));
    return .{
        .gate = .{ .trellis = st[gate_t], .suh = st[suh13], .svh = st[svh_g] },
        .up = .{ .trellis = st[up_t], .suh = up_suh, .svh = st[svh_u] },
        .down = .{ .trellis = st[down_t], .suh = st[suh2], .svh = st[svh2] },
    };
}

/// One layer of the pack as a rate-grouped ragged bank (groups by bits, K2 first). The per-group stacks are the only
/// copy kept apart from the ragged storage, and they are freed before this returns.
fn loadLayer(a: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, kortexa: []const u8, layer: usize) !Loaded {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);

    const ck_path = try std.fmt.allocPrint(a, "{s}/checkpoint.json", .{kortexa});
    const ck_text = try std.Io.Dir.cwd().readFileAlloc(io, ck_path, a, .limited(64 << 20));
    const ck = try std.json.parseFromSlice(std.json.Value, a, ck_text, .{});
    defer ck.deinit();
    var key_buf: [16]u8 = undefined;
    const layers = (ck.value.object.get("layers") orelse return error.BadCheckpoint).object;
    const lay = layers.get(try std.fmt.bufPrint(&key_buf, "{d}", .{layer})) orelse return error.BadCheckpoint;
    const batches = (lay.object.get("batches") orelse return error.BadCheckpoint).array.items;

    var held: [E][8]mlx.mlx_array = @splat(@splat(.{ .ctx = null }));
    errdefer for (&held) |*h| freeAll(h);
    var bits: [E]u8 = @splat(0);
    for (batches) |b| {
        const start: usize = @intCast((b.object.get("start") orelse return error.BadCheckpoint).integer);
        const end: usize = @intCast((b.object.get("end") orelse return error.BadCheckpoint).integer);
        const file = (b.object.get("file") orelse return error.BadCheckpoint).string;
        var meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        var map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ kortexa, file }, 0);
        try mlx.check(mlx.mlx_load_safetensors(&map, &meta, path.ptr, cpu));
        // The tensors are named by the expert's index inside the file or by its global id; the first expert tells which.
        var probe: [64]u8 = undefined;
        const global = hasTensor(map, (try std.fmt.bufPrintSentinel(&probe, "experts.{d}.w13.trellis", .{start}, 0)).ptr);
        if (end > E) return error.BadCheckpoint;
        for (start..end) |e| bits[e] = try expertPieces(cpu, map, if (global) e else e - start, &held[e]);
    }
    for (bits) |b| if (b == 0) return error.ExpertsMissing;

    var banks: [max_groups]root.Bank = undefined;
    var made: usize = 0;
    errdefer for (banks[0..made]) |*b| freeBank(b);
    var slot_of: [E]u32 = undefined;
    var members: [E]u32 = undefined;
    var base: u32 = 0;
    for (1..9) |want| {
        var n: usize = 0;
        for (0..E) |e| {
            if (@as(usize, bits[e]) != want) continue;
            members[n] = @intCast(e);
            slot_of[e] = base + @as(u32, @intCast(n));
            n += 1;
        }
        if (n == 0) continue;
        if (made == max_groups) return error.TooManyRateGroups;
        banks[made] = try buildBank(a, s, &held, members[0..n]);
        made += 1;
        base += @intCast(n);
    }

    var rb = try root.Ragged.init(s, banks[0..made]);
    errdefer rb.deinit();
    const storage = rb.storage();
    try evalAll(&storage);
    // The group views are lazy slices of the storage: evaluate them now, so no run pays for them.
    var views: [9 * max_groups]mlx.mlx_array = undefined;
    var nv: usize = 0;
    for (rb.views[0..rb.view_count]) |v| {
        for ([_]root.Proj{ v.gate, v.up, v.down }) |p| {
            views[nv..][0..3].* = .{ p.trellis, p.suh, p.svh };
            nv += 3;
        }
    }
    try evalAll(views[0..nv]);
    for (banks[0..made]) |*b| freeBank(b);
    made = 0;
    _ = mlx.mlx_clear_cache();
    return .{ .rb = rb, .slot_of = slot_of };
}

fn printLoaded(layer: usize, rb: *const root.Ragged, load_s: f64) void {
    var line: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&line);
    w.print("{{\"loaded\": {{\"layer\": {d}, \"groups\": [", .{layer}) catch return;
    for (0..rb.count) |g| {
        w.print("{s}{{\"bits\": {d}, \"experts\": {d}}}", .{ if (g == 0) "" else ", ", rb.rates[g][0] / 16, rb.experts_of[g] }) catch return;
    }
    w.print("], \"load_s\": {d:.1}, \"peak_gib\": {d:.2}}}}}\n", .{ load_s, peakGiB() }) catch return;
    std.debug.print("{s}", .{w.buffered()});
}

// ---- The routes: every sequence's rows of one layer ----

/// The rows of every sequence in the taps dir for one layer, concatenated in sequence order: the MoE input (bf16 bits)
/// and the picks as ragged slots. Sequence q owns rows `start[q]..start[q + 1]`; its generated rows begin at
/// `prompt[q] - 1`, the last prompt row, whose output is the first generated token.
const Routes = struct {
    n: usize,
    prompt: []usize,
    start: []usize,
    x: []u16,
    slots: []u32,

    fn rows(self: *const Routes) usize {
        return self.start[self.n];
    }

    fn deinit(self: *Routes) void {
        const page = std.heap.page_allocator;
        page.free(self.prompt);
        page.free(self.start);
        page.free(self.x);
        page.free(self.slots);
    }
};

fn loadRoutes(io: std.Io, dir: []const u8, xm_slot: usize, layer: usize, slot_of: *const [E]u32) !Routes {
    const page = std.heap.page_allocator;
    var prompt: std.ArrayList(usize) = .empty;
    errdefer prompt.deinit(page);
    var start: std.ArrayList(usize) = .empty;
    errdefer start.deinit(page);
    var xs: std.ArrayList(u16) = .empty;
    errdefer xs.deinit(page);
    var slots: std.ArrayList(u32) = .empty;
    errdefer slots.deinit(page);
    try start.append(page, 0);
    var seq: usize = 0;
    while (true) : (seq += 1) {
        const meta_path = try std.fmt.allocPrint(page, "{s}/seq-{d:0>5}.tokens.json", .{ dir, seq });
        defer page.free(meta_path);
        const meta = std.Io.Dir.cwd().readFileAlloc(io, meta_path, page, .limited(64 << 20)) catch |e| switch (e) {
            error.FileNotFound => break,
            else => return e,
        };
        defer page.free(meta);
        const parsed = try std.json.parseFromSlice(std.json.Value, page, meta, .{});
        defer parsed.deinit();
        const p: usize = @intCast((parsed.value.object.get("prompt") orelse return error.BadTokens).integer);
        if (p == 0) return error.BadTokens;

        const eids_path = try std.fmt.allocPrint(page, "{s}/seq-{d:0>5}.eids.bin", .{ dir, seq });
        defer page.free(eids_path);
        const eids_bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, eids_path, page, .limited(1 << 30), .of(u32), null);
        defer page.free(eids_bytes);
        const eids = std.mem.bytesAsSlice(u32, eids_bytes);
        if (eids.len == 0 or eids.len % (MODEL_LAYERS * K) != 0) return error.BadEids;
        const T = eids.len / (MODEL_LAYERS * K);

        const xm_path = try std.fmt.allocPrint(page, "{s}/seq-{d:0>5}.xm.bin", .{ dir, seq });
        defer page.free(xm_path);
        const xm_bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, xm_path, page, .limited(1 << 31), .of(u16), null);
        defer page.free(xm_bytes);
        const xm = std.mem.bytesAsSlice(u16, xm_bytes);
        if (xm.len != T * 2 * H or p > T) return error.BadXm;

        try xs.ensureUnusedCapacity(page, T * H);
        try slots.ensureUnusedCapacity(page, T * K);
        for (0..T) |r| {
            xs.appendSliceAssumeCapacity(xm[(r * 2 + xm_slot) * H ..][0..H]);
            for (eids[(r * MODEL_LAYERS + layer) * K ..][0..K]) |e| {
                if (e >= E) return error.BadExpertId;
                slots.appendAssumeCapacity(slot_of[e]);
            }
        }
        try prompt.append(page, p);
        try start.append(page, start.items[start.items.len - 1] + T);
    }
    if (seq == 0) return error.NoTaps;
    return .{
        .n = seq,
        .prompt = try prompt.toOwnedSlice(page),
        .start = try start.toOwnedSlice(page),
        .x = try xs.toOwnedSlice(page),
        .slots = try slots.toOwnedSlice(page),
    };
}

/// The first row of window `k` of `w` consecutive generated rows: sequences in turn, each at its own place in the run.
fn windowRow(r: *const Routes, w: usize, k: usize) !usize {
    const q = k % r.n;
    const first = r.prompt[q] - 1;
    const generated = r.start[q + 1] - r.start[q] - first;
    if (generated < w) return error.SequenceTooShort;
    return r.start[q] + first + (k * 53 + 17) % (generated - w + 1);
}

fn distinctOf(slots: []const u32) usize {
    var seen: [E]bool = @splat(false);
    var n: usize = 0;
    for (slots) |sl| {
        if (!seen[sl]) n += 1;
        seen[sl] = true;
    }
    return n;
}

// ---- Decode and prefill: `moeRagged` over rows of the routes ----

/// Rows `lo .. lo + rows` of the routes as one call's inputs: x [1, rows, H] bf16, picks [1, rows, K] u32, scores 1/8.
const Chunk = struct {
    x: mlx.mlx_array,
    inds: mlx.mlx_array,
    scores: mlx.mlx_array,

    fn init(r: *const Routes, lo: usize, rows: usize) !Chunk {
        if (lo + rows > r.rows()) return error.TooFewRows;
        return .{
            .x = upload(u16, r.x[lo * H ..][0 .. rows * H], &.{ 1, rows, H }, .bfloat16),
            .inds = upload(u32, r.slots[lo * K ..][0 .. rows * K], &.{ 1, rows, K }, .uint32),
            .scores = upload(f32, eighth[0 .. rows * K], &.{ 1, rows, K }, .float32),
        };
    }

    fn deinit(self: Chunk) void {
        free(self.x);
        free(self.inds);
        free(self.scores);
    }

    /// The routed experts of these rows over `x`: the chunk's own, or any array of its shape.
    fn run(self: Chunk, s: mlx.mlx_stream, rb: *const root.Ragged, x: mlx.mlx_array) !mlx.mlx_array {
        return root.moeRagged(s, x, rb, self.inds, self.scores, mul1, false, null, null);
    }
};

/// The `CALLS` windows of `w` consecutive generated rows one decode measurement chains, one per call.
const WindowSet = struct {
    chunks: [CALLS]Chunk,
    /// Mean distinct experts a call picks.
    distinct: f64,

    fn init(r: *const Routes, w: usize) !WindowSet {
        var set: WindowSet = undefined;
        var distinct: usize = 0;
        for (0..CALLS) |k| {
            const row = try windowRow(r, w, k);
            set.chunks[k] = try Chunk.init(r, row, w);
            distinct += distinctOf(r.slots[row * K ..][0 .. w * K]);
        }
        set.distinct = @as(f64, @floatFromInt(distinct)) / CALLS;
        return set;
    }

    fn deinit(self: *const WindowSet) void {
        for (self.chunks) |c| c.deinit();
    }
};

/// Median microseconds per call of a `CALLS`-call dependent chain (each output * 0 + the next window's x feeds the next
/// call), fresh routes per call: the E1 method over real rows.
fn chainTime(s: mlx.mlx_stream, io: std.Io, rb: *const root.Ragged, set: *const WindowSet, zero: mlx.mlx_array) !f64 {
    var samples: [ROUNDS]u64 = undefined;
    var round: usize = 0;
    while (round < ROUNDS + WARMUPS) : (round += 1) {
        var sw = io_util.Stopwatch.init(io);
        var y = try op(mlx.mlx_add, .{ set.chunks[0].x, zero, s });
        for (set.chunks, 0..) |chunk, k| {
            const out = try chunk.run(s, rb, y);
            const z = try op(mlx.mlx_multiply, .{ out, zero, s });
            free(out);
            const next = try op(mlx.mlx_add, .{ z, set.chunks[(k + 1) % CALLS].x, s });
            free(z);
            free(y);
            y = next;
        }
        try mlx.check(mlx.mlx_array_eval(y));
        free(y);
        const ns = sw.read();
        if (round >= WARMUPS) samples[round - WARMUPS] = ns / CALLS;
    }
    return @as(f64, @floatFromInt(median(&samples))) / 1000.0;
}

fn decodeTiming(s: mlx.mlx_stream, io: std.Io, rb: *const root.Ragged, routes: *const Routes, layer: usize, zero: mlx.mlx_array) !void {
    for (WIDTHS) |w| {
        const set = try WindowSet.init(routes, w);
        defer set.deinit();
        const us = try chainTime(s, io, rb, &set, zero);
        std.debug.print("{{\"decode\": {{\"layer\": {d}, \"rows\": {d}, \"us\": {d:.1}, \"distinct\": {d:.1}}}}}\n", .{ layer, w, us, set.distinct });
    }
}

/// The bf16 elements of a reference output that are NaN or infinite, and the largest magnitude among the rest: a bit
/// comparison of NaNs or zeros would pass for nothing, so the bits lines carry these.
const Scan = struct {
    nonfinite: usize = 0,
    absmax: f32 = 0,

    fn add(self: *Scan, bytes: []const u8) void {
        for (std.mem.bytesAsSlice(u16, bytes)) |b| {
            if (b & 0x7F80 == 0x7F80) {
                self.nonfinite += 1;
            } else {
                self.absmax = @max(self.absmax, @abs(bf16ToF32(b)));
            }
        }
    }
};

/// A window's output against each of its rows run alone: the rows whose bytes differ.
fn decodeBits(s: mlx.mlx_stream, rb: *const root.Ragged, routes: *const Routes, layer: usize) !void {
    for (BIT_WIDTHS) |w| {
        var differ: usize = 0;
        var scan: Scan = .{};
        for (0..BIT_WINDOWS) |k| {
            const row = try windowRow(routes, w, k);
            const window = try Chunk.init(routes, row, w);
            defer window.deinit();
            const together = try window.run(s, rb, window.x);
            defer free(together);
            var held = mlx.mlx_array_new();
            defer free(held);
            const window_bytes = try bytesOf(s, together, &held);
            scan.add(window_bytes);
            for (0..w) |i| {
                const single = try Chunk.init(routes, row + i, 1);
                defer single.deinit();
                const alone = try single.run(s, rb, single.x);
                defer free(alone);
                var held1 = mlx.mlx_array_new();
                defer free(held1);
                const row_bytes = try bytesOf(s, alone, &held1);
                if (!std.mem.eql(u8, window_bytes[i * H * 2 ..][0 .. H * 2], row_bytes)) differ += 1;
            }
        }
        std.debug.print("{{\"decode_bits\": {{\"layer\": {d}, \"rows\": {d}, \"differ\": {d}, \"of\": {d}, \"nonfinite\": {d}, \"absmax\": {d:.3}}}}}\n", .{
            layer, w, differ, BIT_WINDOWS * w, scan.nonfinite, scan.absmax,
        });
    }
}

fn prefillTiming(s: mlx.mlx_stream, io: std.Io, rb: *const root.Ragged, routes: *const Routes, layer: usize) !void {
    for (CHUNKS) |c| {
        const chunk = try Chunk.init(routes, 0, c);
        defer chunk.deinit();
        var samples: [PREFILL_ROUNDS]u64 = undefined;
        var round: usize = 0;
        // The first call warms the kernels and is not timed.
        while (round < PREFILL_ROUNDS + 1) : (round += 1) {
            var sw = io_util.Stopwatch.init(io);
            const out = try chunk.run(s, rb, chunk.x);
            defer free(out);
            try mlx.check(mlx.mlx_array_eval(out));
            const ns = sw.read();
            if (round > 0) samples[round - 1] = ns;
        }
        const ns = median(&samples);
        std.debug.print("{{\"prefill\": {{\"layer\": {d}, \"rows\": {d}, \"ms\": {d:.2}, \"tok_s\": {d:.0}}}}}\n", .{
            layer, c, @as(f64, @floatFromInt(ns)) / 1e6, @as(f64, @floatFromInt(c)) / seconds(ns),
        });
    }
}

/// The largest chunk's output against the same rows run as two halves of 2048 and as eight pieces of 512: the rows whose
/// bytes differ.
fn prefillBits(s: mlx.mlx_stream, rb: *const root.Ragged, routes: *const Routes, layer: usize) !void {
    const total = CHUNKS[CHUNKS.len - 1];
    const all = try Chunk.init(routes, 0, total);
    defer all.deinit();
    const whole = try all.run(s, rb, all.x);
    defer free(whole);
    var held = mlx.mlx_array_new();
    defer free(held);
    const whole_bytes = try bytesOf(s, whole, &held);
    var scan: Scan = .{};
    scan.add(whole_bytes);
    for ([_]usize{ 2048, 512 }) |part| {
        var differ: usize = 0;
        for (0..total / part) |p| {
            const chunk = try Chunk.init(routes, p * part, part);
            defer chunk.deinit();
            const piece = try chunk.run(s, rb, chunk.x);
            defer free(piece);
            var held_piece = mlx.mlx_array_new();
            defer free(held_piece);
            const piece_bytes = try bytesOf(s, piece, &held_piece);
            for (0..part) |r| {
                if (!std.mem.eql(u8, whole_bytes[(p * part + r) * H * 2 ..][0 .. H * 2], piece_bytes[r * H * 2 ..][0 .. H * 2])) differ += 1;
            }
        }
        std.debug.print("{{\"prefill_bits\": {{\"layer\": {d}, \"partition\": \"{d}x{d}\", \"differ\": {d}, \"of\": {d}, \"nonfinite\": {d}, \"absmax\": {d:.3}}}}}\n", .{
            layer, total / part, part, differ, total, scan.nonfinite, scan.absmax,
        });
    }
}

// ---- Precision ----

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn refPath(a: std.mem.Allocator, cfg: Config, layer: usize) ![]u8 {
    return std.fmt.allocPrint(a, "{s}/ref-l{d}.act.bf16", .{ cfg.out, layer });
}

fn writeOut(io: std.Io, cfg: Config, layer: usize, suffix: []const u8, bytes: []const u8) !void {
    const page = std.heap.page_allocator;
    const path = try std.fmt.allocPrint(page, "{s}/sushi-l{d}.{s}", .{ cfg.out, layer, suffix });
    defer page.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

/// Rows `row0 .. row0 + 64` of sequence `seq`, eight picks a row in pick order, each on its own group's bank:
///   - the gate and up GEMVs' f32 planes before the output Hadamard (two input halves a slot), through `pairGemv`;
///   - the down projection of the reference's bf16 SwiGLU activations (as f16), through `projectIndexed`: the f16-store
///     proxy for Sushi's w2.
fn precision(s: mlx.mlx_stream, io: std.Io, cfg: Config, loaded: *const Loaded, routes: *const Routes, layer: usize) !void {
    const page = std.heap.page_allocator;
    if (cfg.seq >= routes.n) return error.PrecisionSequenceMissing;
    const seq_rows = routes.start[cfg.seq + 1] - routes.start[cfg.seq];
    if (cfg.row0 + PRECISION_ROWS > seq_rows) return error.PrecisionRowsPastEnd;
    const first = routes.start[cfg.seq] + cfg.row0;
    const S = PRECISION_ROWS * K;

    const act_path = try refPath(page, cfg, layer);
    defer page.free(act_path);
    const act_bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, act_path, page, .limited(1 << 28), .of(u16), null);
    defer page.free(act_bytes);
    if (act_bytes.len != S * I * 2) return error.BadActSize;
    const act = std.mem.bytesAsSlice(u16, act_bytes);
    // bf16 to f16 is exact unless a value falls below f16's normal range; count the ones that move.
    var inexact: usize = 0;
    for (act) |b| {
        const v = bf16ToF32(b);
        const h: f16 = @floatCast(v);
        if (@as(f32, h) != v) inexact += 1;
    }

    const g_out = try page.alloc(f32, S * 2 * I);
    defer page.free(g_out);
    const u_out = try page.alloc(f32, S * 2 * I);
    defer page.free(u_out);
    const y_out = try page.alloc(f16, S * H);
    defer page.free(y_out);

    kernels.setDecodeParams(mul1);
    const rb = &loaded.rb;
    for (0..rb.count) |g| {
        var picks: std.ArrayList(usize) = .empty;
        defer picks.deinit(page);
        for (0..S) |j| {
            if (rb.groupOf(routes.slots[first * K + j]) == g) try picks.append(page, j);
        }
        const n = picks.items.len;
        if (n == 0) continue;
        const xg = try page.alloc(u16, n * H);
        defer page.free(xg);
        const ag = try page.alloc(u16, n * I);
        defer page.free(ag);
        const ids = try page.alloc(u32, n);
        defer page.free(ids);
        for (picks.items, 0..) |j, i| {
            @memcpy(xg[i * H ..][0..H], routes.x[(first + j / K) * H ..][0..H]);
            @memcpy(ag[i * I ..][0..I], act[j * I ..][0..I]);
            ids[i] = routes.slots[first * K + j] - rb.base[g];
        }
        const x = upload(u16, xg, &.{ n, H }, .bfloat16);
        defer free(x);
        const local = upload(u32, ids, &.{n}, .uint32);
        defer free(local);
        const bank = rb.views[g];

        const pair = try kernels.pairGemv(s, x, bank.gate.suh, bank.gate.suh, bank.gate.trellis, bank.up.trellis, local, @intCast(H), @intCast(I), @intCast(n), 1, 0);
        defer free(pair[0]);
        defer free(pair[1]);
        try evalAll(&.{ pair[0], pair[1] });
        const gp = mlx.mlx_array_data_float32(pair[0]) orelse return error.Unreadable;
        const up = mlx.mlx_array_data_float32(pair[1]) orelse return error.Unreadable;
        for (picks.items, 0..) |j, i| {
            @memcpy(g_out[j * 2 * I ..][0 .. 2 * I], gp[i * 2 * I ..][0 .. 2 * I]);
            @memcpy(u_out[j * 2 * I ..][0 .. 2 * I], up[i * 2 * I ..][0 .. 2 * I]);
        }

        const act_bf16 = upload(u16, ag, &.{ n, I }, .bfloat16);
        defer free(act_bf16);
        const act_f16 = try op(mlx.mlx_astype, .{ act_bf16, .float16, s });
        defer free(act_f16);
        const y = try kernels.projectIndexed(s, act_f16, bank.down.trellis, bank.down.suh, bank.down.svh, local);
        defer free(y);
        try mlx.check(mlx.mlx_array_eval(y));
        const yp = mlx.mlx_array_data_float16(y) orelse return error.Unreadable;
        for (picks.items, 0..) |j, i| @memcpy(y_out[j * H ..][0..H], yp[i * H ..][0..H]);
    }
    try writeOut(io, cfg, layer, "g.f32", std.mem.sliceAsBytes(g_out));
    try writeOut(io, cfg, layer, "u.f32", std.mem.sliceAsBytes(u_out));
    try writeOut(io, cfg, layer, "y.f16", std.mem.sliceAsBytes(y_out));
    std.debug.print("{{\"precision\": {{\"layer\": {d}, \"seq\": {d}, \"row0\": {d}, \"slots\": {d}, \"act_f16_inexact\": {d}}}}}\n", .{ layer, cfg.seq, cfg.row0, S, inexact });
}

// ---- The test ----

fn printDevice() void {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_get_default_device(&dev) != 0) return;
    defer _ = mlx.mlx_device_free(dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return;
    var arch: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&arch, info, "architecture") != 0) return;
    std.debug.print("{{\"device\": {{\"architecture\": \"{s}\"}}}}\n", .{std.mem.span(arch)});
}

fn runLayer(s: mlx.mlx_stream, io: std.Io, cfg: Config, layer: usize, xm_slot: usize) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    _ = mlx.mlx_reset_peak_memory();
    var sw = io_util.Stopwatch.init(io);
    var loaded = try loadLayer(arena.allocator(), io, s, cfg.kortexa, layer);
    defer loaded.deinit();
    printLoaded(layer, &loaded.rb, seconds(sw.read()));
    var routes = try loadRoutes(io, cfg.taps, xm_slot, layer, &loaded.slot_of);
    defer routes.deinit();

    const zero0 = mlx.mlx_array_new_float(0);
    defer free(zero0);
    const zero = try op(mlx.mlx_astype, .{ zero0, .bfloat16, s });
    defer free(zero);
    if (cfg.legs.time) try decodeTiming(s, io, &loaded.rb, &routes, layer, zero);
    if (cfg.legs.bits) try decodeBits(s, &loaded.rb, &routes, layer);
    if (cfg.legs.time) try prefillTiming(s, io, &loaded.rb, &routes, layer);
    if (cfg.legs.bits) try prefillBits(s, &loaded.rb, &routes, layer);
    if (cfg.legs.prec) try precision(s, io, cfg, &loaded, &routes, layer);
}

test "inkling2 real layers: Sushi decode, prefill, bits and precision" {
    if (!on()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    // The harness's observable output includes the engaged-arm lines (`[exl3] ... funnel engaged arm=nax|simdmat`).
    log.enableStderr();
    // The runner has printed the test's name without a newline: start the JSON lines on a fresh one.
    std.debug.print("\n", .{});
    const io = std.testing.io;
    const cfg = try config();
    try std.Io.Dir.cwd().createDirPath(io, cfg.out);
    printDevice();
    std.debug.print("{{\"legs\": {{\"time\": {}, \"bits\": {}, \"prec\": {}}}}}\n", .{ cfg.legs.time, cfg.legs.bits, cfg.legs.prec });
    // The precision leg needs the FP64 reference's activations: fail before the long work, not after it.
    if (cfg.legs.prec) for (REAL_LAYERS) |l| {
        const path = try refPath(std.heap.page_allocator, cfg, l.id);
        defer std.heap.page_allocator.free(path);
        var f = std.Io.Dir.cwd().openFile(io, path, .{}) catch |e| {
            std.debug.print("{s}: {s} (run `ink2q.tfexperts ref` first)\n", .{ path, @errorName(e) });
            return e;
        };
        f.close(io);
    };
    for (REAL_LAYERS) |l| try runLayer(s, io, cfg, l.id, l.xm);
}
