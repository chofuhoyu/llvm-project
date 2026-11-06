// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implementation of the minimal Skeleton runtime ABI.
//
// The runtime owns three things: the operator table, the device probe cache,
// and the placement decision. It does not own compilation; when the cache has
// no implementation for the chosen device it asks the installed provider,
// which is the driver. Artifacts have no provider, so they can only run the
// implementations that were linked into them.

#include "mlir/Dialect/Skeleton/Runtime/SkeletonRuntime.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

constexpr int32_t kDeviceCount = SKELETON_DEVICE_COUNT;

const char *deviceName(int32_t device) {
  switch (device) {
  case SKELETON_DEVICE_CPU:
    return "CPU";
  case SKELETON_DEVICE_GPU:
    return "GPU";
  default:
    return "?";
  }
}

struct OpEntry {
  bool declared = false;
  int32_t preference = SKELETON_DEVICE_CPU;
  int64_t bytes = 0;
  void *impls[kDeviceCount] = {};
  int64_t resolves = 0;
  int64_t compiles[kDeviceCount] = {};
};

struct RuntimeState {
  /// Indexed by operator id.
  std::vector<OpEntry> ops;
  skeleton_rt_provider_fn provider = nullptr;

  bool trace = false;
  bool traceEnvRead = false;

  /// -1 means "not forced": fall back to the real probe.
  int32_t forcedAvailable[kDeviceCount] = {-1, -1};
  int64_t forcedFree[kDeviceCount] = {-1, -1};

  int64_t probeRefreshMs = 10;
  bool probeRefreshEnvRead = false;

  bool probeValid[kDeviceCount] = {false, false};
  int32_t probeResult[kDeviceCount] = {0, 0};
  int64_t probeFree[kDeviceCount] = {-1, -1};
  std::chrono::steady_clock::time_point probeStamp[kDeviceCount];

  /// Memory the runtime handed out, keyed by pointer. `managed` records whether
  /// it came from cuMemAllocManaged, because a GPU request that had no driver
  /// to satisfy it falls back to host memory and must be released with free.
  struct Allocation {
    int32_t device;
    bool managed;
  };
  std::unordered_map<void *, Allocation> allocations;
};

RuntimeState &state() {
  static RuntimeState s;
  return s;
}

bool traceEnabled() {
  RuntimeState &s = state();
  if (!s.traceEnvRead) {
    s.traceEnvRead = true;
    const char *env = std::getenv("SKELETON_TRACE");
    if (env && env[0] != '\0' && std::strcmp(env, "0") != 0)
      s.trace = true;
  }
  return s.trace;
}

int64_t probeRefreshMs() {
  RuntimeState &s = state();
  if (!s.probeRefreshEnvRead) {
    s.probeRefreshEnvRead = true;
    if (const char *env = std::getenv("SKELETON_PROBE_REFRESH_MS")) {
      char *end = nullptr;
      long long value = std::strtoll(env, &end, 10);
      if (end != env && value >= 0)
        s.probeRefreshMs = value;
    }
  }
  return s.probeRefreshMs;
}

/// CUDA driver entry points. The driver API is opened with dlopen so that a
/// machine without a driver never has to resolve against libcuda.
struct CudaDriver {
  bool initialized = false;
  bool available = false;
  int (*init)(unsigned) = nullptr;
  int (*getCount)(int *) = nullptr;
  int (*ctxGetCurrent)(void **) = nullptr;
  int (*memGetInfo)(size_t *, size_t *) = nullptr;
  int (*ctxSynchronize)() = nullptr;
  int (*ctxCreate)(void **, unsigned, int) = nullptr;
  int (*deviceGet)(int *, int) = nullptr;
  int (*primaryCtxRetain)(void **, int) = nullptr;
  int (*ctxPushCurrent)(void *) = nullptr;
  int (*ctxPopCurrent)(void **) = nullptr;
  int (*memAllocManaged)(void **, size_t, unsigned) = nullptr;
  int (*memFree)(unsigned long long) = nullptr;
  /// The retained primary context. Retained once and never released: the
  /// modules the runtime hands memory to keep using it.
  void *primaryContext = nullptr;
  /// A context the runtime created for its own driver calls.
  void *privateContext = nullptr;
  int device = 0;
  int contextError = 0;
  bool contextReady = false;
};

CudaDriver &cudaDriver() {
  static CudaDriver driver;
  if (driver.initialized)
    return driver;
  driver.initialized = true;
#if defined(_WIN32)
  HMODULE handle = LoadLibraryA("nvcuda.dll");
  if (handle) {
    driver.init = (int (*)(unsigned))GetProcAddress(handle, "cuInit");
    driver.getCount =
        (int (*)(int *))GetProcAddress(handle, "cuDeviceGetCount");
    driver.ctxGetCurrent =
        (int (*)(void **))GetProcAddress(handle, "cuCtxGetCurrent");
    driver.memGetInfo =
        (int (*)(size_t *, size_t *))GetProcAddress(handle, "cuMemGetInfo");
    driver.ctxSynchronize =
        (int (*)())GetProcAddress(handle, "cuCtxSynchronize");
    driver.deviceGet = (int (*)(int *, int))GetProcAddress(handle, "cuDeviceGet");
    driver.ctxCreate =
        (int (*)(void **, unsigned, int))GetProcAddress(handle, "cuCtxCreate_v2");
    driver.primaryCtxRetain = (int (*)(void **, int))GetProcAddress(
        handle, "cuDevicePrimaryCtxRetain");
    driver.ctxPushCurrent =
        (int (*)(void *))GetProcAddress(handle, "cuCtxPushCurrent");
    driver.ctxPopCurrent =
        (int (*)(void **))GetProcAddress(handle, "cuCtxPopCurrent");
    driver.memAllocManaged = (int (*)(void **, size_t, unsigned))
        GetProcAddress(handle, "cuMemAllocManaged");
    driver.memFree =
        (int (*)(unsigned long long))GetProcAddress(handle, "cuMemFree_v2");
  }
#else
  void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    handle = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
  if (handle) {
    driver.init = (int (*)(unsigned))dlsym(handle, "cuInit");
    driver.getCount = (int (*)(int *))dlsym(handle, "cuDeviceGetCount");
    driver.ctxGetCurrent = (int (*)(void **))dlsym(handle, "cuCtxGetCurrent");
    driver.memGetInfo =
        (int (*)(size_t *, size_t *))dlsym(handle, "cuMemGetInfo");
    driver.ctxSynchronize = (int (*)())dlsym(handle, "cuCtxSynchronize");
    driver.deviceGet = (int (*)(int *, int))dlsym(handle, "cuDeviceGet");
    driver.ctxCreate =
        (int (*)(void **, unsigned, int))dlsym(handle, "cuCtxCreate_v2");
    driver.primaryCtxRetain =
        (int (*)(void **, int))dlsym(handle, "cuDevicePrimaryCtxRetain");
    driver.ctxPushCurrent = (int (*)(void *))dlsym(handle, "cuCtxPushCurrent");
    driver.ctxPopCurrent = (int (*)(void **))dlsym(handle, "cuCtxPopCurrent");
    driver.memAllocManaged =
        (int (*)(void **, size_t, unsigned))dlsym(handle, "cuMemAllocManaged");
    // The driver exports the versioned names; the unversioned ones are
    // compatibility entry points with a different ABI.
    driver.memFree =
        (int (*)(unsigned long long))dlsym(handle, "cuMemFree_v2");
    if (!driver.memFree)
      driver.memFree = (int (*)(unsigned long long))dlsym(handle, "cuMemFree");
  }
#endif
  if (driver.init && driver.getCount && driver.init(0) == 0) {
    int count = 0;
    driver.available = driver.getCount(&count) == 0 && count > 0;
  }
  return driver;
}

bool cudaAvailable() { return cudaDriver().available; }

/// Run `fn` with the primary CUDA context of the default device current.
/// Allocating managed memory and synchronizing both need a current context, and
/// the runtime cannot assume the caller left one in place. Returns false when
/// the driver API is missing or the context cannot be made current.
bool withPrimaryContext(const std::function<bool(CudaDriver &)> &fn,
                        bool *pushedOut = nullptr) {
  CudaDriver &driver = cudaDriver();
  if (!driver.available)
    return false;

  // The order matters: the primary context is only usable after the device has
  // been looked up. Retaining it before any cuDeviceGet call leaves the caller
  // without a context, and every allocation then fails with
  // CUDA_ERROR_INVALID_CONTEXT.
  if (!driver.contextReady) {
    driver.contextReady = true;
    if (driver.deviceGet)
      driver.deviceGet(&driver.device, 0);
    // A context of the runtime's own comes first. The machine this was
    // developed on hands out a primary context handle that
    // cuCtxPushCurrent rejects with CUDA_ERROR_INVALID_CONTEXT, and retrying it
    // on every operator would turn a workable situation into thousands of
    // driver errors. Managed memory belongs to the device rather than to the
    // context, so memory allocated here is still visible to the modules the
    // driver loaded.
    if (driver.ctxCreate) {
      driver.ctxCreate(&driver.privateContext, 0, driver.device);
      // The created context is current from here on, which is what the rest of
      // the runtime relies on.
    }
    if (!driver.privateContext && driver.primaryCtxRetain)
      driver.primaryCtxRetain(&driver.primaryContext, driver.device);
  }

  // cuCtxCreate makes its context current, and pushing a context that is
  // already current is an error on the driver this ran against. Only push when
  // there is nothing current.
  void *current = nullptr;
  bool haveCurrent =
      driver.ctxGetCurrent && driver.ctxGetCurrent(&current) == 0 && current;
  void *context =
      driver.privateContext ? driver.privateContext : driver.primaryContext;
  bool pushed = false;
  driver.contextError = 0;
  // Push unless this very context is already current: pushing a context that is
  // current is an error, and any other current context may be a stale handle
  // left behind by a caller that popped what it did not push.
  if (context && driver.ctxPushCurrent && current != context) {
    driver.contextError = driver.ctxPushCurrent(context);
    pushed = driver.contextError == 0;
  }
  if (pushedOut)
    *pushedOut = pushed;
  bool result = fn(driver);
  if (pushed && driver.ctxPopCurrent) {
    void *popped = nullptr;
    driver.ctxPopCurrent(&popped);
  }
  return result;
}

/// A one-line description of the context state, for the failure paths. Getting
/// this wrong is the difference between an allocation that works and one that
/// reports CUDA_ERROR_INVALID_CONTEXT, and the two look identical from outside.
const char *contextStateText() {
  CudaDriver &driver = cudaDriver();
  void *current = nullptr;
  bool haveCurrent =
      driver.ctxGetCurrent && driver.ctxGetCurrent(&current) == 0 && current;
  if (haveCurrent)
    return "a context is current";
  if (driver.privateContext || driver.primaryContext)
    return "a context is held but not current";
  return "no CUDA context";
}

/// Free device memory in bytes, or -1 when it cannot be queried. The driver
/// API needs a current context; before the first GPU allocation there is none,
/// in which case the memory check is skipped.
int64_t cudaFreeBytes() {
  // Only ask when this call made the context current itself. Another component
  // may have popped what it did not push, and a handle that is still reported
  // as current can be one the driver no longer accepts. An unreadable count is
  // a supported answer: the placement decision then skips the memory check
  // rather than guessing.
  bool pushed = false;
  int64_t freeBytes = -1;
  withPrimaryContext(
      [&](CudaDriver &driver) {
        if (!driver.memGetInfo)
          return false;
        size_t free = 0;
        size_t total = 0;
        if (driver.memGetInfo(&free, &total) != 0)
          return false;
        freeBytes = static_cast<int64_t>(free);
        return true;
      },
      &pushed);
  if (!pushed)
    return -1;
  return freeBytes;
}

OpEntry *lookup(int64_t opId) {
  RuntimeState &s = state();
  if (opId < 0)
    return nullptr;
  if (static_cast<size_t>(opId) >= s.ops.size())
    s.ops.resize(static_cast<size_t>(opId) + 1);
  OpEntry &entry = s.ops[static_cast<size_t>(opId)];
  if (!entry.declared)
    return nullptr;
  return &entry;
}

} // namespace

extern "C" {

void skeleton_rt_declare(int64_t opId, int32_t preference, int64_t bytes) {
  RuntimeState &s = state();
  if (opId < 0)
    return;
  if (static_cast<size_t>(opId) >= s.ops.size())
    s.ops.resize(static_cast<size_t>(opId) + 1);
  OpEntry &entry = s.ops[static_cast<size_t>(opId)];
  entry.declared = true;
  entry.preference = (preference >= 0 && preference < kDeviceCount)
                         ? preference
                         : SKELETON_DEVICE_CPU;
  entry.bytes = bytes;
  if (traceEnabled())
    std::fprintf(stderr, "[skeleton-rt] op=%lld declare pref=%s bytes=%lld\n",
                 static_cast<long long>(opId), deviceName(entry.preference),
                 static_cast<long long>(bytes));
}

void skeleton_rt_register(int64_t opId, int32_t device, void *impl) {
  if (opId < 0 || device < 0 || device >= kDeviceCount)
    return;
  RuntimeState &s = state();
  if (static_cast<size_t>(opId) >= s.ops.size())
    s.ops.resize(static_cast<size_t>(opId) + 1);
  s.ops[static_cast<size_t>(opId)].impls[device] = impl;
  if (traceEnabled())
    std::fprintf(stderr, "[skeleton-rt] op=%lld register device=%s impl=%p\n",
                 static_cast<long long>(opId), deviceName(device), impl);
}

void skeleton_rt_sync(int32_t device) {
  if (device != SKELETON_DEVICE_GPU)
    return;
  withPrimaryContext([](CudaDriver &driver) {
    if (!driver.ctxSynchronize)
      return false;
    return driver.ctxSynchronize() == 0;
  });
}

void *skeleton_rt_alloc(int32_t device, int64_t bytes) {
  RuntimeState &s = state();
  if (bytes < 1)
    bytes = 1;

  if (device == SKELETON_DEVICE_GPU && cudaAvailable()) {
    void *ptr = nullptr;
    int error = 0;
    bool allocated = withPrimaryContext([&](CudaDriver &driver) {
      if (!driver.memAllocManaged)
        return false;
      // CU_MEM_ATTACH_GLOBAL: the allocation is readable and writable from
      // both the host and the device.
      error = driver.memAllocManaged(&ptr, static_cast<size_t>(bytes), 1);
      return error == 0 && ptr != nullptr;
    });
    if (!allocated) {
      // There is a device, so handing back host memory would only turn this
      // into a fault inside a kernel later.
      std::fprintf(stderr,
                   "[skeleton-rt] error: cannot allocate %lld bytes of managed "
                   "memory (CUDA error %d; %s; context error %d)\n",
                   static_cast<long long>(bytes), error, contextStateText(),
                   cudaDriver().contextError);
      std::exit(1);
    }
    s.allocations[ptr] = {device, /*managed=*/true};
    return ptr;
  }
  // No device to allocate for: a program that never reaches one still has to
  // run, and an artifact on a machine without CUDA has to allocate as well.

  const size_t alignment = 64;
  size_t size = static_cast<size_t>(bytes);
  size = (size + alignment - 1) / alignment * alignment;
  void *ptr = std::aligned_alloc(alignment, size);
  if (!ptr) {
    std::fprintf(stderr,
                 "[skeleton-rt] error: cannot allocate %lld bytes\n",
                 static_cast<long long>(bytes));
    std::exit(1);
  }
  s.allocations[ptr] = {device, /*managed=*/false};
  return ptr;
}

void skeleton_rt_free(int32_t device, void *ptr) {
  if (!ptr)
    return;
  RuntimeState &s = state();
  auto it = s.allocations.find(ptr);
  if (it == s.allocations.end())
    return;
  bool managed = it->second.managed;
  s.allocations.erase(it);
  if (managed) {
    withPrimaryContext([&](CudaDriver &driver) {
      return driver.memFree &&
             driver.memFree(reinterpret_cast<unsigned long long>(ptr)) == 0;
    });
    return;
  }
  (void)device;
  std::free(ptr);
}

int32_t skeleton_rt_executor_id(void) { return 0; }

int32_t skeleton_rt_executor_count(void) { return 1; }

int32_t skeleton_rt_probe(int32_t device) {
  if (device < 0 || device >= kDeviceCount)
    return 0;

  RuntimeState &s = state();

  if (s.forcedAvailable[device] >= 0) {
    // A forced availability has no real memory reading behind it; only a
    // forced free-byte count gives the placement decision a number.
    s.probeFree[device] = s.forcedFree[device];
    return s.forcedAvailable[device];
  }

  auto now = std::chrono::steady_clock::now();
  if (s.probeValid[device]) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - s.probeStamp[device])
                       .count();
    if (elapsed < probeRefreshMs()) {
      if (s.forcedFree[device] >= 0)
        s.probeFree[device] = s.forcedFree[device];
      return s.probeResult[device];
    }
  }

  int32_t result = 0;
  int64_t freeBytes = -1;
  if (device == SKELETON_DEVICE_CPU) {
    result = 1;
  } else if (device == SKELETON_DEVICE_GPU) {
    result = cudaAvailable() ? 1 : 0;
    if (result)
      freeBytes = cudaFreeBytes();
  }

  s.probeValid[device] = true;
  s.probeResult[device] = result;
  s.probeFree[device] =
      s.forcedFree[device] >= 0 ? s.forcedFree[device] : freeBytes;
  s.probeStamp[device] = now;
  return result;
}

void *skeleton_rt_resolve(int64_t opId) {
  RuntimeState &s = state();
  OpEntry *entry = lookup(opId);
  if (!entry) {
    std::fprintf(stderr, "[skeleton-rt] error: op=%lld was never declared\n",
                 static_cast<long long>(opId));
    std::exit(1);
  }

  ++entry->resolves;

  const int32_t preference = entry->preference;
  int32_t candidates[2];
  unsigned numCandidates = 0;
  candidates[numCandidates++] = preference;
  if (preference != SKELETON_DEVICE_CPU)
    candidates[numCandidates++] = SKELETON_DEVICE_CPU;

  // The decision is reported once, after the loop: `probe` and `free` describe
  // the preferred device, `choose` is where the operator ended up, and
  // `reason` says why it is not the preference ("-" when it is).
  const char *reason = "-";
  int32_t chosen = -1;
  void *impl = nullptr;

  for (unsigned i = 0; i < numCandidates; ++i) {
    int32_t device = candidates[i];
    if (!skeleton_rt_probe(device)) {
      reason = "device-unavailable";
      continue;
    }
    // `bytes` is a lower bound on the operands' footprint. Do not place an
    // operator on a GPU that cannot hold them; falling back to the CPU is
    // better than failing inside the kernel allocation.
    if (device == SKELETON_DEVICE_GPU && entry->bytes > 0 &&
        s.probeFree[device] >= 0 && s.probeFree[device] < entry->bytes) {
      reason = "insufficient-memory";
      continue;
    }
    if (!entry->impls[device] && s.provider) {
      void *compiled = s.provider(opId, device);
      // A provider may declare or register operators, which can grow the table
      // and move every entry; look this one up again before writing to it.
      entry = &s.ops[static_cast<size_t>(opId)];
      if (compiled) {
        entry->impls[device] = compiled;
        ++entry->compiles[device];
        if (traceEnabled())
          std::fprintf(
              stderr, "[skeleton-rt] op=%lld compile device=%s impl=%p\n",
              static_cast<long long>(opId), deviceName(device), compiled);
      }
    }
    if (entry->impls[device]) {
      chosen = device;
      impl = entry->impls[device];
      break;
    }
    reason = "no-implementation";
  }

  if (traceEnabled()) {
    int64_t freeBytes =
        preference == SKELETON_DEVICE_GPU ? s.probeFree[preference] : -1;
    char freeText[32] = "-";
    if (freeBytes >= 0)
      std::snprintf(freeText, sizeof(freeText), "%lldMB",
                    static_cast<long long>(freeBytes / (1024 * 1024)));
    std::fprintf(stderr,
                 "[skeleton-rt] op=%lld pref=%s probe=%s free=%s choose=%s "
                 "reason=%s exec=%d\n",
                 static_cast<long long>(opId), deviceName(preference),
                 deviceName(preference), freeText,
                 chosen < 0 ? "none" : deviceName(chosen), reason,
                 skeleton_rt_executor_id());
  }

  if (!impl) {
    std::fprintf(
        stderr,
        "[skeleton-rt] error: no implementation for op=%lld (pref=%s)\n",
        static_cast<long long>(opId), deviceName(preference));
    std::exit(1);
  }
  return impl;
}

void skeleton_rt_set_device_available(int32_t device, int32_t available) {
  if (device < 0 || device >= kDeviceCount)
    return;
  RuntimeState &s = state();
  s.forcedAvailable[device] = available ? 1 : 0;
  s.probeValid[device] = false;
  if (traceEnabled())
    std::fprintf(stderr, "[skeleton-rt] device=%s available=%d (forced)\n",
                 deviceName(device), available ? 1 : 0);
}

void skeleton_rt_set_free_bytes(int32_t device, int64_t bytes) {
  if (device < 0 || device >= kDeviceCount)
    return;
  RuntimeState &s = state();
  s.forcedFree[device] = bytes;
  s.probeValid[device] = false;
  if (traceEnabled())
    std::fprintf(stderr, "[skeleton-rt] device=%s free_bytes=%lld (forced)\n",
                 deviceName(device), static_cast<long long>(bytes));
}

void skeleton_rt_set_provider(skeleton_rt_provider_fn fn) {
  state().provider = fn;
}

void skeleton_rt_set_trace(int32_t enable) {
  RuntimeState &s = state();
  s.traceEnvRead = true;
  s.trace = enable != 0;
}

void skeleton_rt_dump_stats(void) {
  RuntimeState &s = state();
  for (size_t i = 0; i < s.ops.size(); ++i) {
    const OpEntry &entry = s.ops[i];
    if (!entry.declared)
      continue;
    std::fprintf(
        stderr,
        "[skeleton-rt] stats op=%zu pref=%s bytes=%lld resolves=%lld "
        "cpu_impl=%d gpu_impl=%d cpu_compiles=%lld gpu_compiles=%lld "
        "exec=%d\n",
        i, deviceName(entry.preference), static_cast<long long>(entry.bytes),
        static_cast<long long>(entry.resolves),
        entry.impls[SKELETON_DEVICE_CPU] != nullptr,
        entry.impls[SKELETON_DEVICE_GPU] != nullptr,
        static_cast<long long>(entry.compiles[SKELETON_DEVICE_CPU]),
        static_cast<long long>(entry.compiles[SKELETON_DEVICE_GPU]),
        skeleton_rt_executor_id());
  }
}

} // extern "C"
