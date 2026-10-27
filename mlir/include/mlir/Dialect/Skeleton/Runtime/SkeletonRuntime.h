//===----------------------------------------------------------------------===//
//
// Minimal runtime ABI for Skeleton operator placement.
//
// This header is plain C and must not depend on MLIR: the same library is
// linked into the driver (which owns the compiler) and into generated program
// artifacts (which do not).
//
// The contract is small on purpose. The program declares its operator units
// once at start-up, registers one implementation per (op, device), and then
// goes through `skeleton_rt_resolve` at every operator call site. Resolve
// decides which registered implementation to call, asking the provider (the
// driver) to compile one when the cache has no implementation for the chosen
// device.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SKELETON_RUNTIME_SKELETONRUNTIME_H
#define MLIR_DIALECT_SKELETON_RUNTIME_SKELETONRUNTIME_H

#include <stdint.h>

#if defined(_WIN32)
#define SKELETON_RT_EXPORT __declspec(dllexport)
#else
#define SKELETON_RT_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Devices the runtime knows about. The numeric values are part of the ABI:
/// they are written into generated code and into the op table.
enum SkeletonDevice {
  SKELETON_DEVICE_CPU = 0,
  SKELETON_DEVICE_GPU = 1,
  SKELETON_DEVICE_COUNT = 2
};

/// Called by the runtime when the cache has no implementation for `opId` on
/// `device`. It returns the compiled implementation, or NULL when compiling
/// that (op, device) pair is not possible. The provider is owned by the
/// driver; artifacts have none.
typedef void *(*skeleton_rt_provider_fn)(int64_t opId, int32_t device);

/// Declare an operator unit. `preference` is the placement preference that
/// came from the source annotation (a `SkeletonDevice` value). `bytes` is the
/// size of the operator's data, or 0 when unknown; it is passed to the
/// provider and recorded for experiments, not interpreted by the runtime.
SKELETON_RT_EXPORT void skeleton_rt_declare(int64_t opId, int32_t preference,
                                            int64_t bytes);

/// Register the implementation of `opId` for `device`. Registering again
/// replaces the previous implementation, which is how a recompiled variant
/// takes over.
SKELETON_RT_EXPORT void skeleton_rt_register(int64_t opId, int32_t device,
                                             void *impl);

/// Resolve the implementation to call for `opId`. This is the hot path: it
/// must stay O(1) apart from the (cached) device probe. Returns NULL when no
/// device can run the operator; callers treat that as a hard error.
SKELETON_RT_EXPORT void *skeleton_rt_resolve(int64_t opId);

/// Probe a device. Returns non-zero when the device can execute code. Results
/// are cached for a short interval so that resolve does not issue a driver
/// call per operator invocation.
SKELETON_RT_EXPORT int32_t skeleton_rt_probe(int32_t device);

/// Force a device to be considered available or unavailable. Used by tests and
/// by the contention experiments to simulate a busy device.
SKELETON_RT_EXPORT void skeleton_rt_set_device_available(int32_t device,
                                                         int32_t available);

/// Install the provider that compiles cache misses.
SKELETON_RT_EXPORT void skeleton_rt_set_provider(skeleton_rt_provider_fn fn);

/// Turn decision tracing on (non-zero) or off. Also enabled by the
/// SKELETON_TRACE environment variable.
SKELETON_RT_EXPORT void skeleton_rt_set_trace(int32_t enable);

/// Print the per-operator counters collected so far.
SKELETON_RT_EXPORT void skeleton_rt_dump_stats(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLIR_DIALECT_SKELETON_RUNTIME_SKELETONRUNTIME_H
