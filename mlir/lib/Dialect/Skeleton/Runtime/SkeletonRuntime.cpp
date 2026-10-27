//===----------------------------------------------------------------------===//
//
// Implementation of the minimal Skeleton runtime ABI.
//
// The runtime owns three things: the operator table, the device probe cache,
// and the placement decision. It does not own compilation; when the cache has
// no implementation for the chosen device it asks the installed provider,
// which is the driver. Artifacts have no provider, so they can only run the
// implementations that were linked into them.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Skeleton/Runtime/SkeletonRuntime.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
  std::vector<OpEntry> ops;
  skeleton_rt_provider_fn provider = nullptr;

  bool trace = false;
  bool traceEnvRead = false;

  /// -1 means "not forced": fall back to the real probe.
  int32_t forcedAvailable[kDeviceCount] = {-1, -1};

  int64_t probeRefreshMs = 10;
  bool probeRefreshEnvRead = false;

  bool probeValid[kDeviceCount] = {false, false};
  int32_t probeResult[kDeviceCount] = {0, 0};
  std::chrono::steady_clock::time_point probeStamp[kDeviceCount];
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

/// CUDA presence probe. The driver API is opened with dlopen so that a machine
/// without a driver never has to resolve against libcuda.
bool cudaAvailable() {
#if defined(_WIN32)
  static bool initialized = false;
  static bool available = false;
  if (!initialized) {
    initialized = true;
    HMODULE handle = LoadLibraryA("nvcuda.dll");
    if (handle) {
      auto init = (int (*)(unsigned))GetProcAddress(handle, "cuInit");
      auto getCount =
          (int (*)(int *))GetProcAddress(handle, "cuDeviceGetCount");
      if (init && getCount && init(0) == 0) {
        int count = 0;
        available = getCount(&count) == 0 && count > 0;
      }
    }
  }
  return available;
#else
  static bool initialized = false;
  static bool available = false;
  if (!initialized) {
    initialized = true;
    void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      handle = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (handle) {
      auto init = (int (*)(unsigned))dlsym(handle, "cuInit");
      auto getCount =
          (int (*)(int *))dlsym(handle, "cuDeviceGetCount");
      if (init && getCount && init(0) == 0) {
        int count = 0;
        available = getCount(&count) == 0 && count > 0;
      }
    }
  }
  return available;
#endif
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
  entry.preference =
      (preference >= 0 && preference < kDeviceCount) ? preference
                                                     : SKELETON_DEVICE_CPU;
  entry.bytes = bytes;
  if (traceEnabled())
    std::fprintf(stderr,
                 "[skeleton-rt] op=%lld declare pref=%s bytes=%lld\n",
                 static_cast<long long>(opId), deviceName(entry.preference),
                 static_cast<long long>(bytes));
}

void skeleton_rt_register(int64_t opId, int32_t device, void *impl) {
  if (device < 0 || device >= kDeviceCount)
    return;
  RuntimeState &s = state();
  if (static_cast<size_t>(opId) >= s.ops.size())
    s.ops.resize(static_cast<size_t>(opId) + 1);
  s.ops[static_cast<size_t>(opId)].impls[device] = impl;
  if (traceEnabled())
    std::fprintf(stderr, "[skeleton-rt] op=%lld register device=%s impl=%p\n",
                 static_cast<long long>(opId), deviceName(device), impl);
}

int32_t skeleton_rt_probe(int32_t device) {
  if (device < 0 || device >= kDeviceCount)
    return 0;

  RuntimeState &s = state();

  if (s.forcedAvailable[device] >= 0)
    return s.forcedAvailable[device];

  auto now = std::chrono::steady_clock::now();
  if (s.probeValid[device]) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - s.probeStamp[device])
                       .count();
    if (elapsed < probeRefreshMs())
      return s.probeResult[device];
  }

  int32_t result = 0;
  if (device == SKELETON_DEVICE_CPU)
    result = 1;
  else if (device == SKELETON_DEVICE_GPU)
    result = cudaAvailable() ? 1 : 0;

  s.probeValid[device] = true;
  s.probeResult[device] = result;
  s.probeStamp[device] = now;
  return result;
}

void *skeleton_rt_resolve(int64_t opId) {
  RuntimeState &s = state();
  OpEntry *entry = lookup(opId);
  if (!entry) {
    std::fprintf(stderr, "[skeleton-rt] error: op=%lld was never declared\n",
                 static_cast<long long>(opId));
    return nullptr;
  }

  ++entry->resolves;

  int32_t candidates[2];
  unsigned numCandidates = 0;
  candidates[numCandidates++] = entry->preference;
  if (entry->preference != SKELETON_DEVICE_CPU)
    candidates[numCandidates++] = SKELETON_DEVICE_CPU;

  const char *reason = "no-implementation";
  for (unsigned i = 0; i < numCandidates; ++i) {
    int32_t device = candidates[i];
    if (!skeleton_rt_probe(device)) {
      reason = "device-unavailable";
      continue;
    }
    if (!entry->impls[device] && s.provider) {
      void *impl = s.provider(opId, device);
      if (impl) {
        entry->impls[device] = impl;
        ++entry->compiles[device];
        if (traceEnabled())
          std::fprintf(stderr,
                       "[skeleton-rt] op=%lld compile device=%s impl=%p\n",
                       static_cast<long long>(opId), deviceName(device), impl);
      }
    }
    if (entry->impls[device]) {
      if (traceEnabled())
        std::fprintf(stderr,
                     "[skeleton-rt] op=%lld pref=%s probe=%s choose=%s\n",
                     static_cast<long long>(opId),
                     deviceName(entry->preference), deviceName(device),
                     deviceName(device));
      return entry->impls[device];
    }
    reason = "no-implementation";
  }

  if (traceEnabled())
    std::fprintf(stderr,
                 "[skeleton-rt] op=%lld pref=%s choose=none reason=%s\n",
                 static_cast<long long>(opId), deviceName(entry->preference),
                 reason);
  std::fprintf(stderr,
               "[skeleton-rt] error: no implementation for op=%lld (pref=%s)\n",
               static_cast<long long>(opId), deviceName(entry->preference));
  return nullptr;
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
    std::fprintf(stderr,
                 "[skeleton-rt] stats op=%zu pref=%s resolves=%lld "
                 "cpu_impl=%d gpu_impl=%d cpu_compiles=%lld gpu_compiles=%lld\n",
                 i, deviceName(entry.preference),
                 static_cast<long long>(entry.resolves),
                 entry.impls[SKELETON_DEVICE_CPU] != nullptr,
                 entry.impls[SKELETON_DEVICE_GPU] != nullptr,
                 static_cast<long long>(entry.compiles[SKELETON_DEVICE_CPU]),
                 static_cast<long long>(entry.compiles[SKELETON_DEVICE_GPU]));
  }
}

} // extern "C"
