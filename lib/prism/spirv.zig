//! Target-agnostic SPIR-V front end: parses a SPIR-V binary into Vulcan IR.
//! Drivers lower the IR to their own target (NVIDIA SASS, aarch64 JIT).
//! Per-target lowering lives in each driver, not here.

const std = @import("std");
const spirv = @import("vulcan-spirv");
const ir = @import("vulcan-ir");

/// A parsed shader as Vulcan IR. Caller-owned. Deinit when done.
pub const Function = ir.function.Function;

/// The SPIR-V opcode constants and binary builder, re-exported so consumers that
/// only depend on `prism` (e.g. the Vulkan ICD's tests) can assemble SPIR-V test
/// kernels without a direct `vulcan-spirv` dependency.
pub const opcodes = spirv.opcodes;
pub const binary = spirv.binary;

pub const Error = error{InvalidSpirv} || std.mem.Allocator.Error || spirv.lower.Error || spirv.inline_calls.Error;

/// Parse a SPIR-V binary (its entry function) into Vulcan IR. `code` is the raw
/// SPIR-V byte stream: a non-empty multiple of 4 bytes, little-endian u32 words.
/// The returned Function is caller-owned. Deinit it when done.
///
/// Real-world glslang output is normalized first (see `normalize`): function-local
/// scalar variables are promoted to SSA (mem2reg) and module-level storage buffers
/// are reordered into ascending Binding order, so the IR the backend sees matches
/// what `spirv.lowerModule` expects (no Function-storage allocas, buffer parameter
/// order == descriptor binding order).
///
/// Graphics SPIR-V is handled entirely by `vulcan-spirv`. There is no Prism-side rewrite. The lowering consumes glslang's standard shapes natively: per-component
/// vertex-input access (`OpAccessChain` into an Input vector + `OpLoad` of a scalar),
/// gl_Position via the `gl_PerVertex` interface block (an Output `Block` struct whose
/// member is BuiltIn Position, written through `OpAccessChain` + `OpStore`), and UBO
/// matrices (`OpTypeMatrix` + `OpMatrixTimesVector`, fetched element-wise from the
/// buffer in std140 column-major layout). Prism's only front-end job here is SSA
/// construction (mem2reg, below), which is legitimately the front end's responsibility
/// because Vulcan's IR is SSA-only by design.
pub fn parseSpirv(gpa: std.mem.Allocator, code: []const u8) Error!Function {
    if (code.len == 0 or code.len % 4 != 0) return error.InvalidSpirv;
    // SPIR-V is a u32 word stream. Copy into an aligned buffer (the incoming
    // byte slice may be unaligned, e.g. a sub-slice of a larger blob).
    const words = try gpa.alloc(u32, code.len / 4);
    defer gpa.free(words);
    @memcpy(std.mem.sliceAsBytes(words), code);

    // Inline any OpFunctionCall first (Vulcan's pass): a shader that calls helper functions
    // (e.g. vkcube's fragment shader's `linearToSrgb`) becomes one call-free function, so the
    // SSA construction below and Vulcan's single-function lowering both see one function.
    // (Returns a fresh stream even when there is nothing to inline.)
    const inlined = try spirv.inlineCalls(gpa, words);
    defer gpa.free(inlined);

    const normalized = try normalize(gpa, inlined);
    defer gpa.free(normalized);
    return spirv.lowerModule(gpa, normalized);
}

// GL bottom-left origin vs. Vulkan top-left: GLES content renders upside-down without
// a correction. Negate gl_Position.y in every vertex shader's SPIR-V so the GLES path
// renders right-side-up through the same HAL drivers the Vulkan ICD uses.
// Handles both the gl_PerVertex block pattern and the direct BuiltIn Position variable.
// Cull winding is re-inverted in gles.zig to compensate.

/// Return a fresh SPIR-V byte stream identical to `code` except that every store to
/// gl_Position negates the Y component (for the GL bottom-left framebuffer origin).
/// Caller owns the result. If the module has no gl_Position store (not a vertex
/// shader, or malformed) a verbatim copy is returned. Never errors on a well-formed
/// vertex shader. On a parse problem it returns a verbatim copy (the draw still works,
/// just unflipped) rather than failing the link.
pub fn flipPositionY(gpa: std.mem.Allocator, code: []const u8) Error![]u8 {
    if (code.len == 0 or code.len % 4 != 0) return error.InvalidSpirv;
    const words = try gpa.alloc(u32, code.len / 4);
    defer gpa.free(words);
    @memcpy(std.mem.sliceAsBytes(words), code);

    const op = spirv.opcodes;
    const r0 = spirv.binary.Reader.init(words) catch return dupBytes(gpa, code);
    var id_bound = r0.header.id_bound;

    // ---- Locate the gl_Position pointers (the targets an OpStore writes). ----
    // Pattern A: an Output Variable decorated BuiltIn Position -> a direct store target.
    // Pattern B: a struct member 0 decorated BuiltIn Position -> the OpAccessChain into
    //            an Output variable of that struct yields the store target.
    var builtin_pos_vars = std.AutoHashMapUnmanaged(u32, void).empty;
    defer builtin_pos_vars.deinit(gpa);
    var pos_member_structs = std.AutoHashMapUnmanaged(u32, void).empty;
    defer pos_member_structs.deinit(gpa);
    {
        var r = spirv.binary.Reader.init(words) catch return dupBytes(gpa, code);
        r.pos = 5;
        while (true) {
            const inst = (r.next() catch return dupBytes(gpa, code)) orelse break;
            switch (inst.opcode) {
                op.Decorate => if (inst.operands.len >= 3 and
                    inst.operands[1] == op.Decoration.builtin and
                    inst.operands[2] == op.BuiltIn.position)
                {
                    try builtin_pos_vars.put(gpa, inst.operands[0], {});
                },
                op.MemberDecorate => if (inst.operands.len >= 4 and
                    inst.operands[2] == op.Decoration.builtin and
                    inst.operands[3] == op.BuiltIn.position)
                {
                    try pos_member_structs.put(gpa, inst.operands[0], {});
                },
                else => {},
            }
        }
    }

    // Resolve TypePointer pointee + the gl_PerVertex Output variables (pointer to a
    // Position-member struct), then the OpAccessChains into them (member 0). Their
    // result ids are also store targets (the block pattern).
    var ptr_pointee = std.AutoHashMapUnmanaged(u32, u32).empty;
    defer ptr_pointee.deinit(gpa);
    {
        var r = spirv.binary.Reader.init(words) catch return dupBytes(gpa, code);
        r.pos = 5;
        while (true) {
            const inst = (r.next() catch return dupBytes(gpa, code)) orelse break;
            if (inst.opcode == op.TypePointer and inst.operands.len >= 3) {
                try ptr_pointee.put(gpa, inst.operands[0], inst.operands[2]);
            }
        }
    }
    var pervertex_vars = std.AutoHashMapUnmanaged(u32, void).empty;
    defer pervertex_vars.deinit(gpa);
    {
        var r = spirv.binary.Reader.init(words) catch return dupBytes(gpa, code);
        r.pos = 5;
        while (true) {
            const inst = (r.next() catch return dupBytes(gpa, code)) orelse break;
            if (inst.opcode == op.Variable and inst.operands.len >= 3 and inst.operands[2] == op.StorageClass.output) {
                if (ptr_pointee.get(inst.operands[0])) |pointee| {
                    if (pos_member_structs.contains(pointee)) try pervertex_vars.put(gpa, inst.operands[1], {});
                }
            }
        }
    }
    // Integer constants whose value is 0 (the gl_PerVertex member-0 access-chain index is an
    // OpConstant id, not a literal, so resolve which ids stand for 0).
    var zero_const_ids = std.AutoHashMapUnmanaged(u32, void).empty;
    defer zero_const_ids.deinit(gpa);
    {
        var r = spirv.binary.Reader.init(words) catch return dupBytes(gpa, code);
        r.pos = 5;
        while (true) {
            const inst = (r.next() catch return dupBytes(gpa, code)) orelse break;
            // OpConstant: result_type, result, value. A 32-bit integer 0 is value word 0.
            if (inst.opcode == op.Constant and inst.operands.len >= 3 and inst.operands[2] == 0) {
                try zero_const_ids.put(gpa, inst.operands[1], {});
            }
        }
    }

    // Store targets: the direct builtin-Position vars + the access-chains into member 0
    // of a gl_PerVertex var.
    var store_targets = std.AutoHashMapUnmanaged(u32, void).empty;
    defer store_targets.deinit(gpa);
    {
        var it = builtin_pos_vars.keyIterator();
        while (it.next()) |k| try store_targets.put(gpa, k.*, {});
    }
    {
        var r = spirv.binary.Reader.init(words) catch return dupBytes(gpa, code);
        r.pos = 5;
        while (true) {
            const inst = (r.next() catch return dupBytes(gpa, code)) orelse break;
            if (inst.opcode == op.AccessChain and inst.operands.len >= 4) {
                // operands: result_type, result, base, index0... The member index is an
                // OpConstant id. member 0 is the gl_Position member of gl_PerVertex.
                if (pervertex_vars.contains(inst.operands[2]) and zero_const_ids.contains(inst.operands[3])) {
                    try store_targets.put(gpa, inst.operands[1], {});
                }
            }
        }
    }

    if (store_targets.count() == 0) return dupBytes(gpa, code); // not a vertex shader / no Position store

    // ---- Find or synthesize the vec4<f32> type + the constant vec4(1,-1,1,1). ----
    // Reuse existing TypeFloat(32) / TypeVector(float,4) / float constants when present
    // so we don't add redundant declarations. Synthesize what is missing with fresh ids.
    var float32_ty: u32 = 0;
    var vec4_ty: u32 = 0;
    var const_one: u32 = 0;
    var const_neg_one: u32 = 0;
    const f_one: u32 = @bitCast(@as(f32, 1.0));
    const f_neg_one: u32 = @bitCast(@as(f32, -1.0));
    // The byte index (in `words`) just after the last type/constant declaration, where new
    // type/constant decls are inserted (they must precede the function). We track the end of
    // the contiguous global-declaration region: insert right before the first OpFunction.
    var insert_at: usize = words.len;
    {
        var r = spirv.binary.Reader.init(words) catch return dupBytes(gpa, code);
        r.pos = 5;
        while (true) {
            const start = r.pos;
            const inst = (r.next() catch return dupBytes(gpa, code)) orelse break;
            switch (inst.opcode) {
                op.TypeFloat => if (inst.operands.len >= 2 and inst.operands[1] == 32) {
                    float32_ty = inst.operands[0];
                },
                op.TypeVector => if (inst.operands.len >= 3 and inst.operands[2] == 4 and float32_ty != 0 and inst.operands[1] == float32_ty) {
                    vec4_ty = inst.operands[0];
                },
                op.Constant => if (inst.operands.len >= 3 and float32_ty != 0 and inst.operands[0] == float32_ty) {
                    if (inst.operands[2] == f_one) const_one = inst.operands[1];
                    if (inst.operands[2] == f_neg_one) const_neg_one = inst.operands[1];
                },
                op.Function => {
                    insert_at = start;
                    break;
                },
                else => {},
            }
        }
    }
    if (insert_at == words.len) return dupBytes(gpa, code); // no function body

    // Allocate fresh ids for whatever is missing.
    var new_decls: std.ArrayListUnmanaged(u32) = .empty;
    defer new_decls.deinit(gpa);
    const freshId = struct {
        fn next(b: *u32) u32 {
            const id = b.*;
            b.* += 1;
            return id;
        }
    }.next;
    if (float32_ty == 0) {
        float32_ty = freshId(&id_bound);
        try appendInst(gpa, &new_decls, op.TypeFloat, &.{ float32_ty, 32 });
    }
    if (vec4_ty == 0) {
        vec4_ty = freshId(&id_bound);
        try appendInst(gpa, &new_decls, op.TypeVector, &.{ vec4_ty, float32_ty, 4 });
    }
    if (const_one == 0) {
        const_one = freshId(&id_bound);
        try appendInst(gpa, &new_decls, op.Constant, &.{ float32_ty, const_one, f_one });
    }
    if (const_neg_one == 0) {
        const_neg_one = freshId(&id_bound);
        try appendInst(gpa, &new_decls, op.Constant, &.{ float32_ty, const_neg_one, f_neg_one });
    }
    // The flip multiplier constant vec4(1,-1,1,1).
    const flip_const = freshId(&id_bound);
    try appendInst(gpa, &new_decls, op.ConstantComposite, &.{ vec4_ty, flip_const, const_one, const_neg_one, const_one, const_one });

    // ---- Rebuild the word stream: header (with updated id-bound), the prefix up to
    // the first function, the new type/constant decls, then the body with each
    // `OpStore <pos_ptr> <val>` rewritten to multiply the value by the flip constant. ----
    var out: std.ArrayListUnmanaged(u32) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, words[0..5]);
    out.items[3] = id_bound; // updated id bound
    try out.appendSlice(gpa, words[5..insert_at]);
    try out.appendSlice(gpa, new_decls.items);

    {
        var r = spirv.binary.Reader.init(words) catch return dupBytes(gpa, code);
        r.pos = insert_at;
        while (true) {
            const start = r.pos;
            if (start >= words.len) break;
            const head = words[start];
            const wc: usize = head >> 16;
            const inst = (r.next() catch return dupBytes(gpa, code)) orelse break;
            if (inst.opcode == op.Store and inst.operands.len >= 2 and store_targets.contains(inst.operands[0])) {
                const ptr = inst.operands[0];
                const val = inst.operands[1];
                const flipped = freshId(&id_bound);
                try appendInst(gpa, &out, op.FMul, &.{ vec4_ty, flipped, val, flip_const });
                // The (possibly multi-operand) Store, with the value replaced by the flipped id.
                try out.append(gpa, head);
                try out.append(gpa, ptr);
                try out.append(gpa, flipped);
                // Preserve any trailing memory-operand words on the Store.
                if (wc > 3) try out.appendSlice(gpa, words[start + 3 .. start + wc]);
            } else {
                try out.appendSlice(gpa, words[start .. start + wc]);
            }
        }
    }
    out.items[3] = id_bound; // FMul ids bumped the bound further

    const out_words = try out.toOwnedSlice(gpa);
    defer gpa.free(out_words);
    const bytes = try gpa.alloc(u8, out_words.len * 4);
    @memcpy(bytes, std.mem.sliceAsBytes(out_words));
    return bytes;
}

/// Append one SPIR-V instruction (computed word count) to a word list.
fn appendInst(gpa: std.mem.Allocator, list: *std.ArrayListUnmanaged(u32), opcode: u16, operands: []const u32) std.mem.Allocator.Error!void {
    const word_count: u32 = @intCast(1 + operands.len);
    try list.append(gpa, (word_count << 16) | opcode);
    try list.appendSlice(gpa, operands);
}

/// A verbatim heap copy of `code` (used when the flip transform has nothing to do).
fn dupBytes(gpa: std.mem.Allocator, code: []const u8) Error![]u8 {
    return gpa.dupe(u8, code);
}

// Multi-block mem2reg: full SSA construction over the CFG.
// glslang emits function-local OpVariables. Vulcan's IR is SSA-only so they must be promoted.
// Inserts OpPhi at the iterated dominance frontier of each def, renames loads to the
// reaching definition. Textbook Cytron et al.

/// One basic block of the (first) function: its label, the instruction-word ranges
/// of its body (between the label and the terminator), its terminator words, and its
/// CFG successors (resolved to block indices).
const Bb = struct {
    label: u32,
    /// Index into `words` of the first body instruction (after the OpLabel).
    body_start: usize,
    /// Index into `words` just past the last body instruction (at the terminator).
    body_end: usize,
    /// Index into `words` of the terminator instruction's first word.
    term_start: usize,
    /// Index into `words` just past the terminator.
    term_end: usize,
    succs: std.ArrayListUnmanaged(u32) = .empty,
    preds: std.ArrayListUnmanaged(u32) = .empty,
    /// Immediate dominator block index (maxInt = none / entry).
    idom: u32 = std.math.maxInt(u32),
    /// Reverse-post-order number (for dominator iteration).
    rpo: u32 = 0,
};

/// Whether an opcode terminates a basic block.
fn isTerminator(opcode: u16) bool {
    const op = spirv.opcodes;
    return switch (opcode) {
        op.Branch, op.BranchConditional, op.Switch, op.Return, op.ReturnValue, op.Unreachable, op.Kill => true,
        else => false,
    };
}

/// Promote scalar function-local variables of a multi-block function body to SSA,
/// inserting `OpPhi` at merge points and renaming loads. Returns a fresh word stream,
/// or (a verbatim alias of) `words` when there is nothing to do (single block, no
/// function, or no promotable variable). The returned slice equals `words` exactly
/// when unchanged, so the caller frees only when `ptr` differs.
fn promoteMultiBlock(gpa: std.mem.Allocator, words: []const u32) Error![]const u32 {
    const op = spirv.opcodes;
    const r0 = spirv.binary.Reader.init(words) catch return error.InvalidSpirv;
    const bound = r0.header.id_bound;

    // ---- Locate the function and split it into basic blocks. ----
    var bbs = std.ArrayListUnmanaged(Bb).empty;
    defer {
        for (bbs.items) |*b| {
            b.succs.deinit(gpa);
            b.preds.deinit(gpa);
        }
        bbs.deinit(gpa);
    }

    // Function-storage variable ids -> their declaring instruction words.
    var local_vars = std.AutoHashMapUnmanaged(u32, void).empty;
    defer local_vars.deinit(gpa);
    // var id -> its result-type id (the pointer type) so we can read the pointee.
    var var_ptr_type = try gpa.alloc(u32, bound);
    defer gpa.free(var_ptr_type);
    @memset(var_ptr_type, 0);
    // pointer type id -> pointee type id.
    var ptr_pointee = try gpa.alloc(u32, bound);
    defer gpa.free(ptr_pointee);
    @memset(ptr_pointee, 0);

    var first_label: usize = 0; // index of the first OpLabel inside the function

    {
        var r = spirv.binary.Reader.init(words) catch return error.InvalidSpirv;
        r.pos = 5;
        var in_func = false;
        var cur: ?usize = null;
        var seen_label = false;
        while (true) {
            const start = r.pos;
            const inst = (r.next() catch return error.InvalidSpirv) orelse break;
            switch (inst.opcode) {
                op.TypePointer => if (inst.operands.len >= 3) {
                    ptr_pointee[inst.operands[0]] = inst.operands[2];
                },
                op.Function => if (!in_func) {
                    in_func = true;
                },
                op.FunctionEnd => if (in_func) break,
                op.Variable => if (in_func and inst.operands.len >= 3 and inst.operands[2] == op.StorageClass.function) {
                    try local_vars.put(gpa, inst.operands[1], {});
                    var_ptr_type[inst.operands[1]] = inst.operands[0];
                } else if (!in_func and inst.operands.len >= 1) {},
                op.Label => if (in_func) {
                    if (!seen_label) {
                        seen_label = true;
                        first_label = start;
                    }
                    // Close the previous block's body at this label.
                    if (cur) |ci| {
                        if (bbs.items[ci].body_end == 0) bbs.items[ci].body_end = start;
                    }
                    try bbs.append(gpa, .{
                        .label = inst.operands[0],
                        .body_start = r.pos,
                        .body_end = 0,
                        .term_start = 0,
                        .term_end = 0,
                    });
                    cur = bbs.items.len - 1;
                },
                else => if (in_func and cur != null and bbs.items[cur.?].term_start == 0 and isTerminator(inst.opcode)) {
                    // The first terminator closes this block's body.
                    bbs.items[cur.?].body_end = start;
                    bbs.items[cur.?].term_start = start;
                    bbs.items[cur.?].term_end = r.pos;
                },
            }
        }
    }

    // Nothing to promote without a multi-block function carrying locals.
    if (bbs.items.len <= 1 or local_vars.count() == 0) return words;

    // Map label id -> block index.
    var label_block = std.AutoHashMapUnmanaged(u32, u32).empty;
    defer label_block.deinit(gpa);
    for (bbs.items, 0..) |b, i| try label_block.put(gpa, b.label, @intCast(i));

    // Resolve successors/predecessors from each block's terminator.
    for (bbs.items, 0..) |*b, bi| {
        if (b.term_start == 0) continue;
        var rr = spirv.binary.Reader.init(words) catch return error.InvalidSpirv;
        rr.pos = b.term_start;
        const inst = (rr.next() catch return error.InvalidSpirv) orelse continue;
        switch (inst.opcode) {
            op.Branch => try addEdge(gpa, bbs.items, &label_block, @intCast(bi), inst.operands[0]),
            op.BranchConditional => {
                try addEdge(gpa, bbs.items, &label_block, @intCast(bi), inst.operands[1]);
                try addEdge(gpa, bbs.items, &label_block, @intCast(bi), inst.operands[2]);
            },
            op.Switch => {
                // OpSwitch %selector %default <lit> %case <lit> %case ...
                // The default and every case label is a successor (so the merge block's phi
                // predecessors are computed correctly). The literal width follows the
                // selector's integer width (1 word for <=32 bits, 2 for 64). Look it up so
                // the (literal, label) stride is right.
                try addEdge(gpa, bbs.items, &label_block, @intCast(bi), inst.operands[1]); // default
                const lit_words: usize = if (selectorBits(words, inst.operands[0]) > 32) 2 else 1;
                const stride = lit_words + 1;
                var k: usize = 2;
                while (k + lit_words < inst.operands.len) : (k += stride) {
                    try addEdge(gpa, bbs.items, &label_block, @intCast(bi), inst.operands[k + lit_words]);
                }
            },
            else => {},
        }
    }

    // ---- Identify promotable variables: scalar, only direct Load/Store uses. ----
    // A variable used by OpAccessChain, OpCopyMemory, as a function argument, etc. has
    // its address taken and cannot be promoted. Drop it from the set.
    var promotable = std.AutoHashMapUnmanaged(u32, void).empty;
    defer promotable.deinit(gpa);
    {
        var it = local_vars.iterator();
        while (it.next()) |e| try promotable.put(gpa, e.key_ptr.*, {});
    }
    {
        var r = spirv.binary.Reader.init(words) catch return error.InvalidSpirv;
        r.pos = first_label;
        while (true) {
            const inst = (r.next() catch return error.InvalidSpirv) orelse break;
            switch (inst.opcode) {
                op.Load => {}, // operands[2] = ptr: a direct use, fine.
                op.Store => {}, // operands[0] = ptr: a direct use, fine.
                // The variable's own declaration names its id but does not use it.
                op.Variable => {},
                op.FunctionEnd => break,
                else => {
                    // Any other instruction referencing a local var's id takes its
                    // address: disqualify it.
                    for (inst.operands) |o| {
                        if (o < bound and promotable.contains(o)) _ = promotable.remove(o);
                    }
                },
            }
        }
    }
    if (promotable.count() == 0) return words;

    // ---- Dominator tree (Cooper-Harvey-Kennedy) over reverse-post-order. ----
    const n: u32 = @intCast(bbs.items.len);
    var order = try gpa.alloc(u32, n); // rpo position -> block index
    defer gpa.free(order);
    {
        // DFS post-order from the entry (block 0), then reverse.
        var visited = try gpa.alloc(bool, n);
        defer gpa.free(visited);
        @memset(visited, false);
        var post = std.ArrayListUnmanaged(u32).empty;
        defer post.deinit(gpa);
        // Iterative DFS with an explicit stack of (block, next-succ-index).
        var stack = std.ArrayListUnmanaged([2]u32).empty;
        defer stack.deinit(gpa);
        try stack.append(gpa, .{ 0, 0 });
        visited[0] = true;
        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            const bi = top[0];
            if (top[1] < bbs.items[bi].succs.items.len) {
                const s = bbs.items[bi].succs.items[top[1]];
                top[1] += 1;
                if (!visited[s]) {
                    visited[s] = true;
                    try stack.append(gpa, .{ s, 0 });
                }
            } else {
                try post.append(gpa, bi);
                _ = stack.pop();
            }
        }
        // order = reverse post-order.
        for (post.items, 0..) |bi, i| order[post.items.len - 1 - i] = bi;
        // Unreachable blocks (not visited) get appended so every block has an rpo slot.
        var fill: u32 = @intCast(post.items.len);
        for (0..n) |bi| if (!visited[bi]) {
            order[fill] = @intCast(bi);
            fill += 1;
        };
        for (order, 0..) |bi, i| bbs.items[bi].rpo = @intCast(i);
    }

    // Iterative dominator computation. idom stored as block index. Entry's idom = itself.
    const idom = try gpa.alloc(u32, n);
    defer gpa.free(idom);
    @memset(idom, std.math.maxInt(u32));
    idom[0] = 0;
    {
        var changed = true;
        while (changed) {
            changed = false;
            // Process in reverse-post-order, skipping the entry.
            for (order) |bi| {
                if (bi == 0) continue;
                if (bbs.items[bi].rpo == std.math.maxInt(u32)) continue;
                var new_idom: u32 = std.math.maxInt(u32);
                for (bbs.items[bi].preds.items) |p| {
                    if (idom[p] == std.math.maxInt(u32)) continue; // pred not yet processed
                    new_idom = if (new_idom == std.math.maxInt(u32)) p else intersect(idom, bbs.items, p, new_idom);
                }
                if (new_idom != std.math.maxInt(u32) and idom[bi] != new_idom) {
                    idom[bi] = new_idom;
                    changed = true;
                }
            }
        }
    }

    // ---- Dominance frontiers. ----
    var df = try gpa.alloc(std.ArrayListUnmanaged(u32), n);
    defer {
        for (df) |*d| d.deinit(gpa);
        gpa.free(df);
    }
    for (df) |*d| d.* = .empty;
    for (0..n) |bi| {
        if (bbs.items[bi].preds.items.len < 2) continue;
        for (bbs.items[bi].preds.items) |p| {
            var runner = p;
            while (runner != idom[bi] and runner != std.math.maxInt(u32)) {
                // Add bi to DF(runner) if absent.
                var present = false;
                for (df[runner].items) |x| if (x == bi) {
                    present = true;
                    break;
                };
                if (!present) try df[runner].append(gpa, @intCast(bi));
                if (idom[runner] == runner) break; // entry: stop
                runner = idom[runner];
            }
        }
    }

    // ---- Collect, per promotable var, the blocks that store to it (def sites) and
    // its pointee (value) type id. ----
    var var_list = std.ArrayListUnmanaged(u32).empty;
    defer var_list.deinit(gpa);
    {
        var it = promotable.iterator();
        while (it.next()) |e| try var_list.append(gpa, e.key_ptr.*);
    }
    // var id -> value type id (pointee of its pointer type).
    var var_val_type = try gpa.alloc(u32, bound);
    defer gpa.free(var_val_type);
    @memset(var_val_type, 0);
    for (var_list.items) |v| var_val_type[v] = ptr_pointee[var_ptr_type[v]];

    // def_blocks[var] = list of block indices that store to it.
    var def_blocks = std.AutoHashMapUnmanaged(u32, std.ArrayListUnmanaged(u32)).empty;
    defer {
        var it = def_blocks.iterator();
        while (it.next()) |e| e.value_ptr.deinit(gpa);
        def_blocks.deinit(gpa);
    }
    for (var_list.items) |v| try def_blocks.put(gpa, v, .empty);
    for (bbs.items, 0..) |b, bi| {
        var rr = spirv.binary.Reader.init(words) catch return error.InvalidSpirv;
        rr.pos = b.body_start;
        while (rr.pos < b.body_end) {
            const inst = (rr.next() catch return error.InvalidSpirv) orelse break;
            if (inst.opcode == op.Store and inst.operands.len >= 1 and promotable.contains(inst.operands[0])) {
                const list = def_blocks.getPtr(inst.operands[0]).?;
                if (list.items.len == 0 or list.items[list.items.len - 1] != bi) {
                    // Avoid duplicate consecutive entries (still fine if a block stores twice).
                    var present = false;
                    for (list.items) |x| if (x == bi) {
                        present = true;
                        break;
                    };
                    if (!present) try list.append(gpa, @intCast(bi));
                }
            }
        }
    }

    // ---- Phi placement: iterated dominance frontier of each var's def sites. ----
    // phi_id[block][var] = the fresh SPIR-V result id of the phi for `var` at `block`
    // (0 = no phi). We also keep a per-block ordered list of (var, phi_id).
    var next_id = bound;
    var phi_at = std.AutoHashMapUnmanaged(u64, u32).empty; // key (block<<32|var) -> phi id
    defer phi_at.deinit(gpa);
    var block_phis = try gpa.alloc(std.ArrayListUnmanaged([2]u32), n); // per block: list of {var, phi_id}
    defer {
        for (block_phis) |*l| l.deinit(gpa);
        gpa.free(block_phis);
    }
    for (block_phis) |*l| l.* = .empty;

    for (var_list.items) |v| {
        const defs = def_blocks.get(v).?;
        // Worklist of blocks needing phi consideration.
        var worklist = std.ArrayListUnmanaged(u32).empty;
        defer worklist.deinit(gpa);
        var has_phi = try gpa.alloc(bool, n);
        defer gpa.free(has_phi);
        @memset(has_phi, false);
        var on_work = try gpa.alloc(bool, n);
        defer gpa.free(on_work);
        @memset(on_work, false);
        for (defs.items) |d| {
            try worklist.append(gpa, d);
            on_work[d] = true;
        }
        while (worklist.items.len > 0) {
            const x = worklist.pop().?;
            for (df[x].items) |y| {
                if (!has_phi[y]) {
                    has_phi[y] = true;
                    const pid = next_id;
                    next_id += 1;
                    try phi_at.put(gpa, (@as(u64, y) << 32) | v, pid);
                    try block_phis[y].append(gpa, .{ v, pid });
                    if (!on_work[y]) {
                        on_work[y] = true;
                        try worklist.append(gpa, y);
                    }
                }
            }
        }
    }

    // ---- Rename: walk the dominator tree, maintaining a reaching-def stack per var.
    // Produce: load-result id -> reaching value id (subst), and per-(block,pred) the
    // phi incoming values. ----
    var subst = std.AutoHashMapUnmanaged(u32, u32).empty; // load result id -> value id
    defer subst.deinit(gpa);
    // phi incoming: key (phi_id<<32 | pred_block) -> value id.
    var phi_in = std.AutoHashMapUnmanaged(u64, u32).empty;
    defer phi_in.deinit(gpa);

    // Children in the dominator tree.
    var dom_children = try gpa.alloc(std.ArrayListUnmanaged(u32), n);
    defer {
        for (dom_children) |*c| c.deinit(gpa);
        gpa.free(dom_children);
    }
    for (dom_children) |*c| c.* = .empty;
    for (0..n) |bi| {
        if (bi == 0) continue;
        if (idom[bi] == std.math.maxInt(u32)) continue; // unreachable
        try dom_children[idom[bi]].append(gpa, @intCast(bi));
    }

    // Per-var reaching-definition stack (value ids). An undefined read yields 0, which
    // we materialize later as an OpUndef-equivalent (a zero constant we synthesize).
    var stacks = std.AutoHashMapUnmanaged(u32, std.ArrayListUnmanaged(u32)).empty;
    defer {
        var it = stacks.iterator();
        while (it.next()) |e| e.value_ptr.deinit(gpa);
        stacks.deinit(gpa);
    }
    for (var_list.items) |v| try stacks.put(gpa, v, .empty);

    // Iterative dom-tree walk. Each frame: (block, child-index, [pushes to undo]).
    const Frame = struct {
        block: u32,
        child_idx: u32,
        // ids whose stacks got a push in this block (to pop on exit), stored as var ids
        // with a count; we instead record a flat list of vars pushed.
        pushed: std.ArrayListUnmanaged(u32),
    };
    var walk = std.ArrayListUnmanaged(Frame).empty;
    defer {
        for (walk.items) |*f| f.pushed.deinit(gpa);
        walk.deinit(gpa);
    }

    // Helper closures are awkward in Zig. Inline the per-block "process" on first visit.
    var processed = try gpa.alloc(bool, n);
    defer gpa.free(processed);
    @memset(processed, false);

    try walk.append(gpa, .{ .block = 0, .child_idx = 0, .pushed = .empty });
    while (walk.items.len > 0) {
        const frame = &walk.items[walk.items.len - 1];
        const bi = frame.block;
        if (!processed[bi]) {
            processed[bi] = true;
            // 1. Phis defined here push their result id as the new reaching def.
            for (block_phis[bi].items) |vp| {
                const v = vp[0];
                const pid = vp[1];
                try stacks.getPtr(v).?.append(gpa, pid);
                try frame.pushed.append(gpa, v);
            }
            // 2. Walk the body: loads -> reaching def (subst). stores -> push new def.
            var rr = spirv.binary.Reader.init(words) catch return error.InvalidSpirv;
            rr.pos = bbs.items[bi].body_start;
            while (rr.pos < bbs.items[bi].body_end) {
                const inst = (rr.next() catch return error.InvalidSpirv) orelse break;
                if (inst.opcode == op.Load and inst.operands.len >= 3 and promotable.contains(inst.operands[2])) {
                    const v = inst.operands[2];
                    const st = stacks.getPtr(v).?;
                    const reaching: u32 = if (st.items.len > 0) st.items[st.items.len - 1] else 0;
                    try subst.put(gpa, inst.operands[1], reaching);
                } else if (inst.opcode == op.Store and inst.operands.len >= 2 and promotable.contains(inst.operands[0])) {
                    const v = inst.operands[0];
                    var val = inst.operands[1];
                    // The stored value may itself be a promoted load result: follow it.
                    if (subst.get(val)) |rv| val = rv;
                    try stacks.getPtr(v).?.append(gpa, val);
                    try frame.pushed.append(gpa, v);
                }
            }
            // 3. For each CFG successor, record the phi incoming value from this block.
            for (bbs.items[bi].succs.items) |s| {
                for (block_phis[s].items) |vp| {
                    const v = vp[0];
                    const pid = vp[1];
                    const st = stacks.getPtr(v).?;
                    const reaching: u32 = if (st.items.len > 0) st.items[st.items.len - 1] else 0;
                    try phi_in.put(gpa, (@as(u64, pid) << 32) | bi, reaching);
                }
            }
        }
        // Descend into the next dom-tree child, else pop and undo pushes.
        if (frame.child_idx < dom_children[bi].items.len) {
            const child = dom_children[bi].items[frame.child_idx];
            frame.child_idx += 1;
            try walk.append(gpa, .{ .block = child, .child_idx = 0, .pushed = .empty });
        } else {
            // Undo this block's pushes.
            for (frame.pushed.items) |v| {
                const st = stacks.getPtr(v).?;
                if (st.items.len > 0) _ = st.pop();
            }
            frame.pushed.deinit(gpa);
            _ = walk.pop();
        }
    }

    // Resolve substitution chains (a promoted load feeding a store feeding another load).
    {
        var it = subst.iterator();
        while (it.next()) |e| {
            var v = e.value_ptr.*;
            var guard: usize = 0;
            while (subst.get(v)) |nv| {
                if (nv == v) break;
                v = nv;
                guard += 1;
                if (guard > subst.count()) break;
            }
            e.value_ptr.* = v;
        }
    }

    // A phi may take a value of 0 (no reaching def on an edge, e.g. a back-edge before
    // the first store, or an unwritten path). Synthesize one zero constant of each
    // needed value type to stand in (the SSA value is dynamically dead on that path).
    var zero_const = std.AutoHashMapUnmanaged(u32, u32).empty; // value-type id -> const id
    defer zero_const.deinit(gpa);
    {
        // Determine which value types appear and need a zero fallback.
        var need_zero = std.AutoHashMapUnmanaged(u32, void).empty;
        defer need_zero.deinit(gpa);
        for (var_list.items) |v| {
            // If any incoming for any of v's phis is 0, we need a zero of v's type.
            const vt = var_val_type[v];
            for (bbs.items, 0..) |_, bi| {
                if (phi_at.get((@as(u64, bi) << 32) | v)) |pid| {
                    for (bbs.items[bi].preds.items) |p| {
                        const incoming = phi_in.get((@as(u64, pid) << 32) | p) orelse 0;
                        if (incoming == 0) {
                            try need_zero.put(gpa, vt, {});
                        }
                    }
                }
            }
        }
        // Also a load that reached no def at all substitutes to 0.
        {
            var it = subst.iterator();
            while (it.next()) |e| if (e.value_ptr.* == 0) {
                // We don't know the type from the load result alone here. Conservatively
                // map all promotable var types. (Loads of a never-written local are rare.)
            };
        }
        var it = need_zero.iterator();
        while (it.next()) |e| {
            const cid = next_id;
            next_id += 1;
            try zero_const.put(gpa, e.key_ptr.*, cid);
        }
    }

    // ---- Emit the rewritten module. ----
    var out = std.ArrayListUnmanaged(u32).empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, words[0..5]);
    out.items[3] = next_id; // new id bound

    const Emit = struct {
        fn one(o: *std.ArrayListUnmanaged(u32), a: std.mem.Allocator, opcode: u16, ops: []const u32) Error!void {
            const wc: u32 = @intCast(1 + ops.len);
            try o.append(a, (wc << 16) | opcode);
            try o.appendSlice(a, ops);
        }
    };

    // Walk the whole module. Outside the function, copy verbatim except: inject the
    // synthesized zero constants just before OpFunction. Inside, rebuild block by block.
    var rmain = spirv.binary.Reader.init(words) catch return error.InvalidSpirv;
    rmain.pos = 5;
    var emitted_zeros = false;
    var in_func2 = false;
    while (true) {
        const start = rmain.pos;
        const inst = (rmain.next() catch return error.InvalidSpirv) orelse break;
        const iw = words[start..rmain.pos];

        if (inst.opcode == op.Function and !in_func2) {
            // Inject zero constants (types are already declared earlier in the module).
            if (!emitted_zeros) {
                emitted_zeros = true;
                var it = zero_const.iterator();
                while (it.next()) |e| {
                    // OpConstant <type> <id> 0  (a 32-bit zero literal; SPIR-V scalars here
                    // are 32-bit ints/floats, one zero word).
                    try Emit.one(&out, gpa, op.Constant, &.{ e.key_ptr.*, e.value_ptr.*, 0 });
                }
            }
            in_func2 = true;
            try out.appendSlice(gpa, iw);
            continue;
        }

        if (!in_func2) {
            try out.appendSlice(gpa, iw);
            continue;
        }

        // Inside the function.
        switch (inst.opcode) {
            // Drop promotable variable declarations.
            op.Variable => {
                if (inst.operands.len >= 2 and promotable.contains(inst.operands[1])) continue;
                try out.appendSlice(gpa, iw);
            },
            op.Label => {
                // Emit the label, then this block's phis at its head.
                try out.appendSlice(gpa, iw);
                const bi = label_block.get(inst.operands[0]).?;
                for (block_phis[bi].items) |vp| {
                    const v = vp[0];
                    const pid = vp[1];
                    const vt = var_val_type[v];
                    // OpPhi: [resultType, resultId, (value, predLabel)*]
                    var phi_ops = std.ArrayListUnmanaged(u32).empty;
                    defer phi_ops.deinit(gpa);
                    try phi_ops.append(gpa, vt);
                    try phi_ops.append(gpa, pid);
                    for (bbs.items[bi].preds.items) |p| {
                        var incoming = phi_in.get((@as(u64, pid) << 32) | p) orelse 0;
                        if (incoming != 0) {
                            // Follow substitution chains for the incoming value too.
                            if (subst.get(incoming)) |rv| incoming = rv;
                        }
                        if (incoming == 0) incoming = zero_const.get(vt) orelse 0;
                        try phi_ops.append(gpa, incoming);
                        try phi_ops.append(gpa, bbs.items[p].label);
                    }
                    try Emit.one(&out, gpa, op.Phi, phi_ops.items);
                }
            },
            op.Load => {
                // Drop promoted loads (their result is substituted at use sites).
                if (inst.operands.len >= 3 and promotable.contains(inst.operands[2])) continue;
                try emitSubstituted(gpa, &out, iw, &subst);
            },
            op.Store => {
                // Drop promoted stores.
                if (inst.operands.len >= 1 and promotable.contains(inst.operands[0])) continue;
                try emitSubstituted(gpa, &out, iw, &subst);
            },
            op.Phi => {
                // Pre-existing phis (rare in glslang Function-var output) pass through
                // with operand substitution.
                try emitSubstituted(gpa, &out, iw, &subst);
            },
            op.FunctionEnd => {
                try out.appendSlice(gpa, iw);
                in_func2 = false;
            },
            else => try emitSubstituted(gpa, &out, iw, &subst),
        }
    }

    return out.toOwnedSlice(gpa);
}

/// The integer bit width of the value `sel_id` (an `OpSwitch` selector), so the switch's
/// (literal, label) operand stride is known (a 64-bit selector has two-word literals).
/// Returns 32 when the type cannot be resolved (the overwhelmingly common case anyway).
fn selectorBits(words: []const u32, sel_id: u32) u32 {
    const op = spirv.opcodes;
    // Find the instruction that defines `sel_id` and read its result-type id (word 1).
    var type_id: u32 = 0;
    var r = spirv.binary.Reader.init(words) catch return 32;
    r.pos = 5;
    while (r.next() catch return 32) |inst| {
        // Skip ops that do not follow the [resultType, resultId, ...] shape, or that name
        // `sel_id` as a non-result operand. Type/constant/label/variable declarations and
        // the value-producing ops we care about (Load, arithmetic, Phi, conversions) all put
        // the result id at operands[1] and the result type at operands[0]. A false positive
        // only yields a wrong width, which falls back to 32 below.
        switch (inst.opcode) {
            op.TypeInt, op.TypeFloat, op.TypeVoid, op.TypeBool, op.TypeVector, op.TypePointer, op.TypeFunction, op.Label, op.Variable, op.Store, op.Branch, op.BranchConditional, op.Switch, op.Return => continue,
            else => {},
        }
        if (inst.operands.len >= 2 and inst.operands[1] == sel_id) {
            type_id = inst.operands[0];
            break;
        }
    }
    if (type_id == 0) return 32;
    // Read that type's width if it is OpTypeInt [result, width, signedness].
    var r2 = spirv.binary.Reader.init(words) catch return 32;
    r2.pos = 5;
    while (r2.next() catch return 32) |inst| {
        if (inst.opcode == op.TypeInt and inst.operands.len >= 2 and inst.operands[0] == type_id) {
            return inst.operands[1];
        }
    }
    return 32;
}

/// Add a CFG edge from block `from` to the block labelled `to_label`.
fn addEdge(gpa: std.mem.Allocator, bbs: []Bb, label_block: *std.AutoHashMapUnmanaged(u32, u32), from: u32, to_label: u32) Error!void {
    const to = label_block.get(to_label) orelse return;
    // Avoid duplicate successor entries (a conditional branch to the same label twice).
    for (bbs[from].succs.items) |s| if (s == to) return;
    try bbs[from].succs.append(gpa, to);
    try bbs[to].preds.append(gpa, from);
}

/// Intersect two nodes in the dominator tree by walking up via `idom` until they meet,
/// comparing by reverse-post-order number (the Cooper-Harvey-Kennedy `intersect`).
fn intersect(idom: []const u32, bbs: []const Bb, a0: u32, b0: u32) u32 {
    var a = a0;
    var b = b0;
    while (a != b) {
        while (bbs[a].rpo > bbs[b].rpo) a = idom[a];
        while (bbs[b].rpo > bbs[a].rpo) b = idom[b];
    }
    return a;
}

/// Append an instruction's words to `out`, replacing any operand id that is a promoted
/// load result with its substituted value (and never the header word). Mirrors
/// `emitWithSubst` but takes the raw word slice directly.
fn emitSubstituted(gpa: std.mem.Allocator, out: *std.ArrayListUnmanaged(u32), iw: []const u32, subst: *std.AutoHashMapUnmanaged(u32, u32)) Error!void {
    try out.append(gpa, iw[0]); // header (count|opcode)
    for (iw[1..]) |w| try out.append(gpa, subst.get(w) orelse w);
}

/// Normalize a raw SPIR-V word stream so the (straight-line) backend can lower it:
///   1. mem2reg: promote single-block function-local scalar `OpVariable`s to SSA.
///      glslang allocates each GLSL local (`uint i = ...`) as a Function-storage
///      variable with OpStore/OpLoad. The lowering models SSA only, so we forward
///      the stored value to every load and drop the variable/store/load words.
///   2. Buffer ordering: reorder module-level storage-buffer / uniform-block
///      `OpVariable`s into ascending `Binding` decoration order. The lowering
///      synthesizes one kernel parameter per buffer in *declaration* order, but a
///      descriptor set binds buffers by *binding index*. Sorting the declarations
///      by binding makes parameter order == binding order.
///
/// Only the straight-line (single basic block) case is handled for mem2reg, which
/// is what a flat compute kernel emits. A variable with loads/stores that a single
/// preceding store does not dominate is left intact (the lowering then rejects it,
/// surfacing as InvalidSpirv, rather than miscompiling).
fn normalize(gpa: std.mem.Allocator, words_in: []const u32) Error![]u32 {
    const op = spirv.opcodes;

    // Multi-block bodies (if/else, for/while) need full SSA construction to promote
    // function-local variables: the value of a local at a merge point depends on which
    // edge reached it, so a single forwarded store does not suffice. `promoteMultiBlock`
    // does the standard mem2reg over the CFG (dominator tree + dominance frontiers + phi
    // insertion + renaming), emitting `OpPhi` at the merge points the backend already
    // lowers. After it runs, every promotable local is gone, so the single-block mem2reg
    // below sees none and only the buffer reordering applies. A single-block body skips
    // promotion (the fast path forwards the lone store directly).
    const promoted = try promoteMultiBlock(gpa, words_in);
    defer if (promoted.ptr != words_in.ptr) gpa.free(@constCast(promoted));
    const words = promoted;

    var r = spirv.binary.Reader.init(words) catch return error.InvalidSpirv;

    // First pass: collect Binding decorations and the function-local variables, and
    // detect whether the (first) function body is a single basic block.
    var binding_of = std.AutoHashMapUnmanaged(u32, u32).empty; // var id -> binding
    defer binding_of.deinit(gpa);
    // Function-local variable ids (declared with StorageClass Function inside the body).
    var local_vars = std.AutoHashMapUnmanaged(u32, void).empty;
    defer local_vars.deinit(gpa);

    {
        var rr = spirv.binary.Reader.init(words) catch return error.InvalidSpirv;
        var in_func = false;
        var label_count: u32 = 0;
        while (rr.next() catch return error.InvalidSpirv) |inst| {
            switch (inst.opcode) {
                op.Decorate => if (inst.operands.len >= 3 and inst.operands[1] == op.Decoration.binding) {
                    try binding_of.put(gpa, inst.operands[0], inst.operands[2]);
                },
                op.Function => in_func = true,
                op.Label => if (in_func) {
                    label_count += 1;
                },
                op.Variable => if (in_func and inst.operands.len >= 3 and inst.operands[2] == op.StorageClass.function) {
                    try local_vars.put(gpa, inst.operands[1], {});
                },
                else => {},
            }
        }
        // mem2reg only when the body is straight-line (a single block). A multi-block
        // body has already had its promotable locals removed by `promoteMultiBlock`.
        // Any locals still present here were not promotable, so leave them intact.
        if (label_count > 1) {
            local_vars.clearRetainingCapacity();
        }
    }

    // Second pass over the body: for each function-local variable, map its loads to
    // the stored SSA value. A single store before the loads (the glslang pattern) is
    // required. If a load precedes the store we bail on that variable.
    var subst = std.AutoHashMapUnmanaged(u32, u32).empty; // load-result id -> stored value id
    defer subst.deinit(gpa);
    var stored = std.AutoHashMapUnmanaged(u32, u32).empty; // var id -> current stored value
    defer stored.deinit(gpa);
    {
        var rr = spirv.binary.Reader.init(words) catch return error.InvalidSpirv;
        while (rr.next() catch return error.InvalidSpirv) |inst| {
            switch (inst.opcode) {
                op.Store => if (inst.operands.len >= 2 and local_vars.contains(inst.operands[0])) {
                    try stored.put(gpa, inst.operands[0], inst.operands[1]);
                },
                op.Load => if (inst.operands.len >= 3 and local_vars.contains(inst.operands[2])) {
                    // result = operands[1], ptr = operands[2]
                    if (stored.get(inst.operands[2])) |val| {
                        try subst.put(gpa, inst.operands[1], val);
                    } else {
                        // Load before any store: cannot promote this var safely.
                        _ = local_vars.remove(inst.operands[2]);
                    }
                },
                else => {},
            }
        }
    }

    // Resolve substitution chains (a promoted load feeding another).
    {
        var it = subst.iterator();
        while (it.next()) |e| {
            var v = e.value_ptr.*;
            var guard: usize = 0;
            while (subst.get(v)) |next_v| {
                v = next_v;
                guard += 1;
                if (guard > subst.count()) break;
            }
            e.value_ptr.* = v;
        }
    }

    // Reordering plan: stable order of storage-buffer/uniform OpVariable ids by
    // ascending binding. We emit the body's module-level buffer OpVariables in this
    // order regardless of their source position.
    var buffer_vars = std.ArrayListUnmanaged(u32).empty;
    defer buffer_vars.deinit(gpa);

    // Rebuild the word stream: header verbatim, then instructions, dropping promoted
    // variable/store/load instructions and substituting load-result references.
    var out = std.ArrayListUnmanaged(u32).empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, words[0..5]); // header

    // Collect the module-level buffer variables and their original instruction words
    // so they can be re-emitted in binding order at the first one's position.
    var buffer_var_words = std.AutoHashMapUnmanaged(u32, []const u32).empty; // var id -> full instr words
    defer buffer_var_words.deinit(gpa);

    r.pos = 5;
    var emitted_buffers = false;
    while (true) {
        const start = r.pos;
        const inst = (r.next() catch return error.InvalidSpirv) orelse break;
        const instr_words = words[start..r.pos];

        // Drop promoted local variable declarations and their stores/loads.
        if (inst.opcode == op.Variable and inst.operands.len >= 2 and local_vars.contains(inst.operands[1])) continue;
        if (inst.opcode == op.Store and inst.operands.len >= 1 and local_vars.contains(inst.operands[0])) continue;
        if (inst.opcode == op.Load and inst.operands.len >= 3 and local_vars.contains(inst.operands[2])) continue;

        // A module-level storage-buffer/uniform/push-constant variable: stash it and emit
        // the whole set, binding-sorted, at the position of the first such variable. A
        // push-constant block carries no Binding decoration, so it sorts AFTER all the
        // descriptor-bound buffers (its slot in the lowered parameter list is therefore the
        // one right after the descriptors), which is the slot the ICD binds its bytes at.
        if (inst.opcode == op.Variable and inst.operands.len >= 3 and
            (inst.operands[2] == op.StorageClass.storage_buffer or inst.operands[2] == op.StorageClass.uniform or inst.operands[2] == op.StorageClass.push_constant))
        {
            const id = inst.operands[1];
            try buffer_var_words.put(gpa, id, instr_words);
            try buffer_vars.append(gpa, id);
            if (!emitted_buffers) {
                // Defer: emit the full sorted set when we hit FunctionEnd-or-later is
                // wrong (vars are before the function). Collect now and emit before the
                // first Function instruction (handled below). We can't know we have all
                // of them yet, but the collect-then-flush approach handles it correctly.
            }
            continue; // skip for now; emitted just before OpFunction
        }

        // Just before the first OpFunction, flush the binding-sorted buffer variables.
        if (inst.opcode == op.Function and !emitted_buffers) {
            try flushBufferVars(gpa, &out, buffer_vars.items, &buffer_var_words, &binding_of);
            emitted_buffers = true;
        }

        // Emit this instruction, substituting any promoted load-result operand refs.
        try emitWithSubst(gpa, &out, inst.opcode, instr_words, &subst);
    }
    if (!emitted_buffers and buffer_vars.items.len > 0) {
        // No function found (malformed). Still emit buffers so output is well-formed.
        try flushBufferVars(gpa, &out, buffer_vars.items, &buffer_var_words, &binding_of);
    }

    return out.toOwnedSlice(gpa);
}

/// Emit the storage-buffer/uniform OpVariable instructions in ascending Binding
/// order (variables without a Binding keep their relative order, after bound ones).
fn flushBufferVars(
    gpa: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u32),
    ids: []const u32,
    var_words: *std.AutoHashMapUnmanaged(u32, []const u32),
    binding_of: *std.AutoHashMapUnmanaged(u32, u32),
) Error!void {
    // Stable selection sort by binding (small N).
    const order = try gpa.alloc(u32, ids.len);
    defer gpa.free(order);
    @memcpy(order, ids);
    var i: usize = 0;
    while (i < order.len) : (i += 1) {
        var min_j = i;
        var j = i + 1;
        while (j < order.len) : (j += 1) {
            const bj = binding_of.get(order[j]) orelse std.math.maxInt(u32);
            const bmin = binding_of.get(order[min_j]) orelse std.math.maxInt(u32);
            if (bj < bmin) min_j = j;
        }
        if (min_j != i) {
            const t = order[i];
            order[i] = order[min_j];
            order[min_j] = t;
        }
    }
    for (order) |id| {
        if (var_words.get(id)) |w| try out.appendSlice(gpa, w);
    }
}

/// Append one instruction's words to `out`, rewriting any operand id that is a
/// promoted load result to its substituted (stored) value. The opcode/word-count
/// header word and the result-type/result-id positions are preserved. Only operand
/// *uses* are substituted. Every word that is a known load-result id is substituted
/// except the instruction's own result id (operand index varies by opcode, but a
/// load-result is never itself a result-id of another instruction after promotion,
/// so a blanket use-substitution is correct here).
fn emitWithSubst(
    gpa: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u32),
    opcode: u16,
    instr_words: []const u32,
    subst: *std.AutoHashMapUnmanaged(u32, u32),
) Error!void {
    _ = opcode;
    try out.append(gpa, instr_words[0]); // header word (count|opcode)
    for (instr_words[1..]) |w| {
        try out.append(gpa, subst.get(w) orelse w);
    }
}

test "parseSpirv lowers a SPIR-V function to Vulcan IR" {
    const gpa = std.testing.allocator;
    const op = spirv.opcodes;
    // int f(int x, int y) { return x*y - x; }
    // Same kernel as Vulcan's own SPIR-V->SASS test. Exercises a real lowering, not a stub.
    var b = try spirv.binary.Builder.init(gpa, 9);
    defer b.deinit(gpa);
    try b.emit(gpa, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1, 1, 1 });
    try b.emit(gpa, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(gpa, op.FunctionParameter, &.{ 1, 4 });
    try b.emit(gpa, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(gpa, op.Label, &.{6});
    try b.emit(gpa, op.IMul, &.{ 1, 7, 4, 5 });
    try b.emit(gpa, op.ISub, &.{ 1, 8, 7, 4 });
    try b.emit(gpa, op.ReturnValue, &.{8});
    try b.emit(gpa, op.FunctionEnd, &.{});

    var func = try parseSpirv(gpa, std.mem.sliceAsBytes(b.words.items));
    defer func.deinit();
    try std.testing.expect(func.blockCount() >= 1);
    try std.testing.expect(func.valueCount() > 0);
}

test "parseSpirv rejects a non-SPIR-V byte stream" {
    const gpa = std.testing.allocator;
    const garbage = [_]u8{ 0xde, 0xad, 0xbe, 0xef } ** 8; // bad magic word
    if (parseSpirv(gpa, &garbage)) |f| {
        var ff = f;
        ff.deinit();
        return error.ShouldHaveRejected;
    } else |_| {}
    try std.testing.expectError(error.InvalidSpirv, parseSpirv(gpa, "abc")); // not a word multiple
}

test "parseSpirv lowers a compute shader that calls a helper (OpFunctionCall inlining)" {
    const gpa = std.testing.allocator;
    // vkfn.comp: a compute shader calling a multi-block branching helper `transform`.
    const raw = @embedFile("testdata_vkfn.spv");
    var func = try parseSpirv(gpa, raw);
    func.deinit();
}

test "parseSpirv lowers vkcube's real fragment shader (inlined linearToSrgb + Pow)" {
    const gpa = std.testing.allocator;
    const raw = @embedFile("testdata_vkcube_fs.spv");
    // Step through so a failure pinpoints inline vs mem2reg vs lowering.
    const words = try gpa.alloc(u32, raw.len / 4);
    defer gpa.free(words);
    @memcpy(std.mem.sliceAsBytes(words), raw);
    const inlined = spirv.inlineCalls(gpa, words) catch |e| {
        std.debug.print("[test] inlineCalls failed: {s}\n", .{@errorName(e)});
        return e;
    };
    defer gpa.free(inlined);
    const normalized = normalize(gpa, inlined) catch |e| {
        std.debug.print("[test] normalize failed: {s}\n", .{@errorName(e)});
        return e;
    };
    defer gpa.free(normalized);
    var func = try spirv.lowerModule(gpa, normalized);
    func.deinit();
}

test "flipPositionY negates gl_Position.y (gl_PerVertex block pattern) and round-trips" {
    const gpa = std.testing.allocator;
    // Embed a real vertex shader (the gradient triangle's vkwin.vert) that writes
    // gl_Position through the gl_PerVertex block (OpAccessChain member 0 + OpStore).
    const raw = @embedFile("testdata_flip_vert.spv");
    const flipped = try flipPositionY(gpa, raw);
    defer gpa.free(flipped);
    // The transform must have changed the stream (an FMul + a flip constant added).
    try std.testing.expect(flipped.len > raw.len);
    // It must contain an OpFMul (opcode 133), the per-component negate.
    const fw = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(flipped)));
    var saw_fmul = false;
    var i: usize = 5;
    while (i < fw.len) {
        const wc: usize = fw[i] >> 16;
        const oc: u16 = @truncate(fw[i] & 0xffff);
        if (wc == 0) break;
        if (oc == spirv.opcodes.FMul) saw_fmul = true;
        i += wc;
    }
    try std.testing.expect(saw_fmul);
    // The flipped SPIR-V still lowers to Vulcan IR (the draw path).
    var func = try parseSpirv(gpa, flipped);
    func.deinit();
}

test "flipPositionY leaves a fragment shader (no gl_Position) verbatim" {
    const gpa = std.testing.allocator;
    const raw = @embedFile("testdata_flip_frag.spv");
    const result = try flipPositionY(gpa, raw);
    defer gpa.free(result);
    try std.testing.expectEqualSlices(u8, raw, result);
}
