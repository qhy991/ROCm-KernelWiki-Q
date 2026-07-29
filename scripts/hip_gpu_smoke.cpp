#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t error = (call);                                             \
        if (error != hipSuccess) {                                             \
            std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error)); \
            return 2;                                                          \
        }                                                                     \
    } while (0)

__global__ void mark_values(int* values, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = index * 2 + 1;
    }
}

int main() {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count != 1) {
        std::fprintf(stderr, "expected one visible GPU, found %d\n", device_count);
        return 3;
    }

    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));

    constexpr int count = 4096;
    int* device_values = nullptr;
    HIP_CHECK(hipMalloc(
        reinterpret_cast<void**>(&device_values),
        count * sizeof(int)
    ));
    hipLaunchKernelGGL(
        mark_values,
        dim3((count + 255) / 256),
        dim3(256),
        0,
        0,
        device_values,
        count
    );
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> host_values(count);
    HIP_CHECK(hipMemcpy(
        host_values.data(),
        device_values,
        count * sizeof(int),
        hipMemcpyDeviceToHost
    ));
    HIP_CHECK(hipFree(device_values));

    for (int index = 0; index < count; ++index) {
        if (host_values[index] != index * 2 + 1) {
            std::fprintf(stderr, "mismatch at %d: %d\n", index, host_values[index]);
            return 4;
        }
    }

    std::printf(
        "PASS visible=1 name=%s arch=%s values=%d\n",
        properties.name,
        properties.gcnArchName,
        count
    );
    return 0;
}
