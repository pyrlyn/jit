#!/bin/bash
# Private copy of the pyrlyn/jit tree for one mutant slot ($MUT_DIR) from the checkout ($MUT_ROOT); the checkout is never written.
set -e
rsync -a --exclude .git --exclude zig-out --exclude .zig-cache --exclude third_party "$MUT_ROOT/" "$MUT_DIR/"
[ -d "$MUT_ROOT/third_party" ] && ln -sfn "$MUT_ROOT/third_party" "$MUT_DIR/third_party"
true
