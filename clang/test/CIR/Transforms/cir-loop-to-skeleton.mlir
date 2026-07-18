// RUN: cir-opt %s -cir-loop-to-skeleton -split-input-file | FileCheck %s

// Test 1: annotated cir.for with preference → skeleton.annotate
module {
  cir.func @test_for_to_annotate() {
    cir.scope {
      cir.for : cond {
        %true = cir.const #true
        cir.condition(%true)
      } body {
        cir.yield
      } step {
        cir.yield
      } [#cir.annotation<"skeleton_vector_add", ["GPU"]>]
    }
    cir.return
  }
}

// CHECK-LABEL: cir.func @test_for_to_annotate
// CHECK: skeleton.annotate preference = #skeleton.preference<"GPU">
// CHECK-SAME: name = "skeleton_vector_add"
// CHECK-NOT: cir.for

// -----

// Test 2: annotated cir.for without preference → default "CPU"
module {
  cir.func @test_for_no_pref() {
    cir.scope {
      cir.for : cond {
        %true = cir.const #true
        cir.condition(%true)
      } body {
        cir.yield
      } step {
        cir.yield
      } [#cir.annotation<"skeleton_vector_add">]
    }
    cir.return
  }
}

// CHECK-LABEL: cir.func @test_for_no_pref
// CHECK: skeleton.annotate preference = #skeleton.preference<"CPU">
// CHECK-SAME: name = "skeleton_vector_add"
// CHECK-NOT: cir.for

// -----

// Test 3: unannotated for loop is left untouched.
module {
  cir.func @test_plain_for() {
    cir.scope {
      cir.for : cond {
        %true = cir.const #true
        cir.condition(%true)
      } body {
        cir.yield
      } step {
        cir.yield
      }
    }
    cir.return
  }
}

// CHECK-LABEL: cir.func @test_plain_for
// CHECK: cir.for
// CHECK-NOT: skeleton.annotate
