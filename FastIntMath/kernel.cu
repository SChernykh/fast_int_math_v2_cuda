#include "fast_int_math_v2.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <Windows.h>

constexpr size_t TEST_SQRT_STEP = 1 << 26;
constexpr size_t TEST_DIV_STEP = 1 << 21;
constexpr int NUMBERS_PER_DIVISOR = 256;

// Run "generate_ptx.bat" and search for "fast_div_heavy BEGIN" in kernel.ptx to look at generated PTX assembly
__global__ void DummyFastDivHeavyPTX(const int64_t* _a, const int32_t* _b, int64_t* _result)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int64_t a = _a[index];
	const int32_t b = _b[index];
	int64_t result;

	asm("// fast_div_heavy BEGIN");
	result = fast_div_heavy(a, b);
	asm("// fast_div_heavy END");

	_result[index] = result;
}

// Run "generate_ptx.bat" and search for "fast_div_v2 BEGIN" in kernel.ptx to look at generated PTX assembly
__global__ void DummyFastDivPTX(const uint64_t* _a, const uint32_t* _b, uint64_t* _result)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t a = _a[index];
	const uint32_t b = _b[index];
	uint64_t result;

	asm("// fast_div_v2 BEGIN");
	result = fast_div_v2(a, b);
	asm("// fast_div_v2 END");

	_result[index] = result;
}

// Run "generate_ptx.bat" and search for "fast_sqrt_v2 BEGIN" in kernel.ptx to look at generated PTX assembly
__global__ void DummyFastSqrtPTX(const uint64_t* _a, uint32_t* _result)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t a = _a[index];
	uint32_t result;

	asm("// fast_sqrt_v2 BEGIN");
	result = fast_sqrt_v2(a);
	asm("// fast_sqrt_v2 END");

	_result[index] = result;
}

__global__ void GetReciprocalTest(const uint32_t base, uint64_t* err_value)
{
	uint32_t b = base + blockIdx.x;
	if (b == 0) b = 1;

	const uint64_t r = get_reciprocal64(b);
	uint64_t r0 = uint64_t(-1) / b;
	double r1 = (uint64_t(-1) % b) + 1;
	r1 /= b;

	const double error = int64_t(r - r0) - r1;

	if (((error <= -0.876) || (error >= 0.876)) && (atomicAdd((uint32_t*)err_value, 1) == 0))
	{
		err_value[1] = b;
		err_value[2] = r;
		err_value[3] = r0;
		err_value[4] = *((uint64_t*)&error);
	}
}

__global__ void FastDivTest(const uint64_t* _a, const uint32_t base, uint64_t* err_value)
{
	const uint64_t a = _a[threadIdx.x];
	uint32_t b = base + blockIdx.x;
	if (b == 0x80000000UL) b = 0x80000001UL;

	const uint64_t fast_div_result = fast_div_v2(a, b);
	const uint64_t correct_result = (uint64_t(a % b) << 32) + (uint32_t)(a / b);
	if ((fast_div_result != correct_result) && (atomicAdd((uint32_t*)err_value, 1) == 0))
	{
		err_value[1] = a;
		err_value[2] = b;
		err_value[3] = fast_div_result;
		err_value[4] = correct_result;
	}
}

__global__ void FastDivHeavyTest(const int64_t* _a, const int32_t base, int64_t* err_value)
{
	const int64_t a = _a[threadIdx.x];
	int32_t b = base + blockIdx.x;
	if (b < 1) b = 1;

	const int64_t fast_div_heavy_result = fast_div_heavy(a, b);
	const int64_t correct_result = a / b;
	if ((fast_div_heavy_result != correct_result) && (atomicAdd((uint32_t*)err_value, 1) == 0))
	{
		err_value[1] = a;
		err_value[2] = b;
		err_value[3] = fast_div_heavy_result;
		err_value[4] = correct_result;
	}
}

__global__ void FastDivHeavyOldTest(const int64_t* _a, const int32_t base, int64_t* err_value)
{
	const int64_t a = _a[threadIdx.x];
	int32_t b = base + blockIdx.x;
	if (b < 2) b = 2;

	const int64_t fast_div_heavy_result = fast_div_heavy_old(a, b);
	const int64_t correct_result = a / b;
	if ((fast_div_heavy_result != correct_result) && (atomicAdd((uint32_t*)err_value, 1) == 0))
	{
		err_value[1] = a;
		err_value[2] = b;
		err_value[3] = fast_div_heavy_result;
		err_value[4] = correct_result;
	}
}

__global__ void FastDivBench(const uint64_t* _a, const uint32_t base, uint64_t* err_value)
{
	const uint64_t a = _a[threadIdx.x];
	uint32_t b = base + blockIdx.x;
	if (b == 0x80000000UL) b = 0x80000001UL;

	const uint64_t fast_div_result = fast_div_v2(a, b);
	if (fast_div_result == 1)
	{
		err_value[0] = 0;
	}
}

__global__ void FastDivHeavyBench(const int64_t* _a, const int32_t base, int64_t* err_value)
{
	const int64_t a = _a[threadIdx.x];
	int32_t b = base + blockIdx.x;
	if (b < 2) b = 2;

	const int64_t fast_div_heavy_result = fast_div_heavy(a, b);
	if (fast_div_heavy_result == 1)
	{
		err_value[0] = 0;
	}
}

__global__ void FastDivHeavyOldBench(const int64_t* _a, const int32_t base, int64_t* err_value)
{
	const int64_t a = _a[threadIdx.x];
	int32_t b = base + blockIdx.x;
	if (b < 2) b = 2;

	const int64_t fast_div_heavy_result = fast_div_heavy_old(a, b);
	if (fast_div_heavy_result == 1)
	{
		err_value[0] = 0;
	}
}

__global__ void NaiveDivHeavyBench(const int64_t* _a, const int32_t base, int64_t* err_value)
{
	const int64_t a = _a[threadIdx.x];
	int32_t b = base + blockIdx.x;
	if (b < 2) b = 2;

	if (a / b == 1)
	{
		err_value[0] = 0;
	}
}

__device__ __forceinline__ void report_sqrt_error(uint64_t i, uint64_t n, uint64_t expected, uint64_t actual, uint64_t* err_value)
{
	if (atomicAdd((uint32_t*)(err_value), 1) == 0)
	{
		err_value[1] = i;
		err_value[2] = n;
		err_value[3] = expected;
		err_value[4] = actual;
	}
}

__global__ void FastSqrtTest(const uint32_t base, uint64_t* err_value)
{
	const int i = base + blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 1779033703)
	{
		if (i == 1779033703)
		{
			const uint64_t n1 = (uint64_t)(-1);
			const uint32_t r1 = fast_sqrt_v2(n1);
			if (r1 != 3558067407U) { report_sqrt_error(i, n1, 3558067407, r1, err_value); }
		}
		return;
	}

	const uint64_t i1 = i + (1ULL << 32);
	const uint64_t n1 = i1 * i1;

	const uint32_t r1 = fast_sqrt_v2(n1);
	if (r1 != i1 * 2 - (1ULL << 33)) { report_sqrt_error(i, n1, i1 * 2 - (1ULL << 33), r1, err_value); }

	const uint32_t r2 = fast_sqrt_v2(n1 + i1);
	if (r2 != i1 * 2 - (1ULL << 33)) { report_sqrt_error(i, n1 + i1, i1 * 2 - (1ULL << 33), r2, err_value); }

	const uint32_t r3 = fast_sqrt_v2(n1 + i1 + 1);
	if (r3 != i1 * 2 + 1 - (1ULL << 33)) { report_sqrt_error(i, n1 + i1 + 1, i1 * 2 + 1 - (1ULL << 33), r2, err_value); }

	const uint64_t i2 = i + (1ULL << 32) + 1;
	const uint64_t n2 = i2 * i2 - 1;

	const uint32_t r4 = fast_sqrt_v2(n2);
	if (r4 != i2 * 2 - 1 - (1ULL << 33)) { report_sqrt_error(i, n2, i2 * 2 - 1 - (1ULL << 33), r4, err_value); }
}

cudaError_t TestIntMath()
{
	int64_t dt;

	uint64_t a[NUMBERS_PER_DIVISOR];
	a[0] = 0;
	a[1] = uint64_t(-1);
	uint64_t k = 11400714819323198485ULL;
	for (int i = 2; i < NUMBERS_PER_DIVISOR; ++i)
	{
		a[i] = k;
		k = k * 2862933555777941757ULL + 3037000493;
	}

	uint64_t* dev_a = nullptr;
	uint64_t* dev_err_value = nullptr;

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, sizeof(uint64_t) * NUMBERS_PER_DIVISOR);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_err_value, sizeof(uint64_t) * 5);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, sizeof(uint64_t) * NUMBERS_PER_DIVISOR, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	printf("Benchmarking fast_div_heavy (all divisors, %d numbers per divisor)\n", NUMBERS_PER_DIVISOR);
	LARGE_INTEGER f;
	QueryPerformanceFrequency(&f);
	dt = 0;
	for (uint64_t base = 0; base < 0x80000000ULL; base += TEST_DIV_STEP * 16)
	{
		printf("%.1f%% done\r", base * 100.0 / 0x80000000ULL);

		LARGE_INTEGER t1, t2;
		QueryPerformanceCounter(&t1);

		FastDivHeavyBench<<<TEST_DIV_STEP * 16, NUMBERS_PER_DIVISOR>>>((int64_t*)dev_a, (int32_t)(base), (int64_t*)dev_err_value);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "FastDivHeavyBench launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		QueryPerformanceCounter(&t2);
		dt += t2.QuadPart - t1.QuadPart;
	}
	printf("100.0%% done, %.3f seconds\n", static_cast<double>(dt) / f.QuadPart);

	printf("Benchmarking fast_div_heavy_old (all divisors, %d numbers per divisor)\n", NUMBERS_PER_DIVISOR);
	dt = 0;
	for (uint64_t base = 0; base < 0x80000000ULL; base += TEST_DIV_STEP * 16)
	{
		printf("%.1f%% done\r", base * 100.0 / 0x80000000ULL);

		LARGE_INTEGER t1, t2;
		QueryPerformanceCounter(&t1);

		FastDivHeavyOldBench<<<TEST_DIV_STEP * 16, NUMBERS_PER_DIVISOR>>> ((int64_t*)dev_a, (int32_t)(base), (int64_t*)dev_err_value);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "FastDivHeavyOldBench launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		QueryPerformanceCounter(&t2);
		dt += t2.QuadPart - t1.QuadPart;
	}
	printf("100.0%% done, %.3f seconds\n", static_cast<double>(dt) / f.QuadPart);

	printf("Benchmarking naive div_heavy (all divisors, %d numbers per divisor)\n", NUMBERS_PER_DIVISOR);
	dt = 0;
	for (uint64_t base = 0; base < 0x80000000ULL; base += TEST_DIV_STEP * 16)
	{
		printf("%.1f%% done\r", base * 100.0 / 0x80000000ULL);

		LARGE_INTEGER t1, t2;
		QueryPerformanceCounter(&t1);

		NaiveDivHeavyBench<<<TEST_DIV_STEP * 16, NUMBERS_PER_DIVISOR>>>((int64_t*)dev_a, (int32_t)(base), (int64_t*)dev_err_value);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "NaiveDivHeavyBench launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		QueryPerformanceCounter(&t2);
		dt += t2.QuadPart - t1.QuadPart;
	}
	printf("100.0%% done, %.3f seconds\n", static_cast<double>(dt) / f.QuadPart);

	printf("Testing fast_div_heavy (all divisors, %d numbers per divisor)\n", NUMBERS_PER_DIVISOR);
	for (uint64_t base = 0; base < 0x80000000ULL; base += TEST_DIV_STEP)
	{
		printf("%.1f%% done\r", base * 100.0 / 0x80000000ULL);

		FastDivHeavyTest<<<TEST_DIV_STEP, NUMBERS_PER_DIVISOR>>>((int64_t*)dev_a, (int32_t)(base), (int64_t*)dev_err_value);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "FastDivHeavyTest launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		uint64_t err_value[5] = {};
		cudaStatus = cudaMemcpy(err_value, dev_err_value, sizeof(uint64_t) * 5, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		if (err_value[0] != 0)
		{
			printf("\nError:\na=%lld\nb=%lld\nfast_div_heavy result=%lld\ncorrect result=%lld\n", err_value[1], err_value[2], err_value[3], err_value[4]);
			goto Error;
		}
	}
	printf("100.0%% done\n");

	printf("Testing fast_div_heavy_old (all divisors, %d numbers per divisor)\n", NUMBERS_PER_DIVISOR);
	for (uint64_t base = 0; base < 0x80000000ULL; base += TEST_DIV_STEP)
	{
		printf("%.1f%% done\r", base * 100.0 / 0x80000000ULL);

		FastDivHeavyOldTest<<<TEST_DIV_STEP, NUMBERS_PER_DIVISOR>>>((int64_t*)dev_a, (int32_t)(base), (int64_t*)dev_err_value);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "FastDivHeavyOldTest launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		uint64_t err_value[5] = {};
		cudaStatus = cudaMemcpy(err_value, dev_err_value, sizeof(uint64_t) * 5, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		if (err_value[0] != 0)
		{
			printf("\nError:\na=%lld\nb=%lld\nfast_div_heavy result=%lld\ncorrect result=%lld\n", err_value[1], err_value[2], err_value[3], err_value[4]);
			goto Error;
		}
	}
	printf("100.0%% done\n");

	printf("Testing get_reciprocal64 (all divisors)\n");
	for (uint64_t base = 0; base < 0x100000000ULL; base += TEST_DIV_STEP)
	{
		printf("%.1f%% done\r", base * 100.0 / 0x100000000ULL);

		GetReciprocalTest<<<TEST_DIV_STEP, 1>>>((uint32_t)(base), dev_err_value);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "GetReciprocalTest launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		uint64_t err_value[5] = {};
		cudaStatus = cudaMemcpy(err_value, dev_err_value, sizeof(uint64_t) * 5, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		if (err_value[0] != 0)
		{
			printf("\nError:\nb=%llu\nr1=%llx\nr2=%llx\ndelta=%.6f", err_value[1], err_value[2], err_value[3], err_value[4]);
			goto Error;
		}
	}
	printf("100.0%% done\n");

	printf("Testing fast_sqrt_v2 (all edge cases)\n");
	for (uint32_t base = 0; base <= 1779033703; base += TEST_SQRT_STEP)
	{
		printf("%.1f%% done\r", base * 100.0 / 1779033703);

		FastSqrtTest<<<TEST_SQRT_STEP / 256, 256>>>(base, dev_err_value);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "FastSqrtTest launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		uint64_t err_value[5] = {};
		cudaStatus = cudaMemcpy(err_value, dev_err_value, sizeof(uint64_t) * 5, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		if (err_value[0] != 0)
		{
			printf("\nFailed for i=%llu, N=%llu: expected %llu, got %llu\n", err_value[1], err_value[2], err_value[3], err_value[4]);
			goto Error;
		}
	}

	printf("Benchmarking fast_div_v2 (all divisors, %d numbers per divisor)\n", NUMBERS_PER_DIVISOR);
	dt = 0;
	for (uint64_t base = 0x80000000ULL; base < 0x100000000ULL; base += TEST_DIV_STEP * 16)
	{
		printf("%.1f%% done\r", (base - 0x80000000ULL) * 100.0 / (0x100000000ULL - 0x80000000ULL));

		LARGE_INTEGER t1, t2;
		QueryPerformanceCounter(&t1);

		FastDivBench<<<TEST_DIV_STEP * 16, NUMBERS_PER_DIVISOR>>>(dev_a, (uint32_t)(base), dev_err_value);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "FastDivTest launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		QueryPerformanceCounter(&t2);
		dt += t2.QuadPart - t1.QuadPart;
	}
	printf("100.0%% done, %.3f seconds\n", static_cast<double>(dt) / f.QuadPart);

	printf("Testing fast_div_v2 (all divisors, %d numbers per divisor)\n", NUMBERS_PER_DIVISOR);
	for (uint64_t base = 0x80000000ULL; base < 0x100000000ULL; base += TEST_DIV_STEP)
	{
		printf("%.1f%% done\r", (base - 0x80000000ULL) * 100.0 / (0x100000000ULL - 0x80000000ULL));

		FastDivTest<<<TEST_DIV_STEP, NUMBERS_PER_DIVISOR>>>(dev_a, (uint32_t)(base), dev_err_value);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "FastDivTest launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		uint64_t err_value[5] = {};
		cudaStatus = cudaMemcpy(err_value, dev_err_value, sizeof(uint64_t) * 5, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		if (err_value[0] != 0)
		{
			printf("\nError:\na=%llu\nb=%llu\nfast_div result=%llu\ncorrect result=%llu\n", err_value[1], err_value[2], err_value[3], err_value[4]);
			goto Error;
		}
	}
	printf("100.0%% done\n");

Error:
	cudaFree(dev_a);
	cudaFree(dev_err_value);

	return cudaStatus;
}

int main()
{
    cudaError_t cudaStatus = TestIntMath();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "TestIntMath failed!");
        return 1;
    }

	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
