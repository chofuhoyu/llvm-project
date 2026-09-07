#!/bin/sh
#
# Peel the `module { ... }` wrapper that cir-opt prints around the
# C++-compiled skeleton functions, so the test's `cat` can join them with the
# driver's bare top-level funcs into a single implicit module. Joining two
# *bare* fragments is a plain merge (that is why the driver is written without
# a `module` wrapper); removing this one wrapper is the only non-trivial step.
#
# The peel checks the two structural assumptions it relies on and exits
# non-zero with a clear message when cir-opt output drifts, instead of
# silently emitting a corrupt module that would fail deep inside mlir-opt with
# a confusing parse error. The assumptions:
#   * the first content line is the module header on a single line
#     `module <name> {` — cir-call-to-skeleton drops the cir.*/dlti.* module
#     attributes, so no `attributes {...}` spills onto further lines;
#   * the last non-blank line is the module's closing `}`.
#
# Usage: skeleton-peel-module.sh INPUT    # prints the module body on stdout
set -eu

input="$1"

header="$(awk 'NF { print; exit }' "$input")"
case "$header" in
  module*'{') ;;
  *)
    echo "skeleton-peel-module.sh: expected the first content line of cir-opt output to be a single-line module header (\"module ... {\"), got:" >&2
    echo "  $header" >&2
    exit 1
    ;;
esac
case "$header" in
  *attributes*)
    echo "skeleton-peel-module.sh: module header carries attributes; the peel does not handle them. cir-call-to-skeleton should have dropped the cir.*/dlti.* module attributes:" >&2
    echo "  $header" >&2
    exit 1
    ;;
esac

awk 'NR > 1 { lines[++n] = $0 }
     END {
       while (n > 0 && lines[n] == "")
         n--;
       if (n == 0 || lines[n] != "}") {
         print "skeleton-peel-module.sh: expected the last non-blank line of cir-opt output to be the module closing \"}\", got:" > "/dev/stderr";
         if (n > 0) print "  " lines[n] > "/dev/stderr";
         exit 1;
       }
       n--; # drop the closing brace
       for (i = 1; i <= n; i++)
         print lines[i]
     }' "$input"
