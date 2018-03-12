#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <intrin.h>
#include <immintrin.h>

#include <chrono>
#include <random>
#include <cstdlib>
#include <cstring>

// _mm256_extractf128_ps for extracting hi part
// _mm256_castps256_ps128 for extracting lo part (no cycles!)

#define EXEC_CNT (1e7)
#define BROADCAST32(fpx) _MM_SHUFFLE(fpx, fpx, fpx, fpx)

#define AVX 0 // set to 0 or 1 (sse3/avx), set appropriate compiler flags and run

#if AVX
#define ALIGN 32
namespace avx
{
#ifdef _IRR_WINDOWS_
	__declspec(align(ALIGN))
#endif
		struct matrix4x3_row
	{
		union
		{
			float m[4][4];
			__m128 row[4];
		};

		inline matrix4x3_row concatenate(const matrix4x3_row& _other)
		{
			_mm256_zeroupper();
			__m256 A01 = _mm256_load_ps(&m[0][0]);
			__m256 A23 = _mm256_load_ps(&m[2][0]);

			matrix4x3_row out;

			_mm256_store_ps(&out.m[0][0], doJob(A01, _other));
			_mm256_store_ps(&out.m[2][0], doJob(A23, _other));

			return out;
		}

		float& operator()(size_t _i, size_t _j) { return m[_i][_j]; }
		const float& operator()(size_t _i, size_t _j) const { return m[_i][_j]; }

	private:
		static inline __m256 doJob(__m256 _A01, const matrix4x3_row& _mtx)
		{
			__m256 res;
			res = _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(0)), _mm256_broadcast_ps(&_mtx.row[0]));
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(1)), _mm256_broadcast_ps(&_mtx.row[1])));
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(2)), _mm256_broadcast_ps(&_mtx.row[2])));
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(3)), _mm256_broadcast_ps(&_mtx.row[3])));
			return res;
		}
	}
#ifndef _IRR_WINDOWS_
	__attribute__((__aligned__(ALIGN)));
#endif
	; // matrix4x3_row

#ifdef _IRR_WINDOWS_
	__declspec(align(ALIGN))
#endif
		struct matrix4x3_col
	{
		union
		{
			float m[4][4];
			__m128 col[4];
		};

		inline matrix4x3_col concatenate(const matrix4x3_col& _other)
		{
			_mm256_zeroupper();
			__m256 A01 = _mm256_load_ps(&m[0][0]);
			__m256 A23 = _mm256_load_ps(&m[2][0]);

			matrix4x3_col out;

			_mm256_store_ps(&out.m[0][0], doJob(A01, _other));
			_mm256_store_ps(&out.m[2][0], doJob(A23, _other));

			return out;
		}

		float& operator()(size_t _i, size_t _j) { return m[_j][_i]; }
		const float& operator()(size_t _i, size_t _j) const { return m[_j][_i]; }

	private:
		static inline __m256 doJob(__m256 _A01, const matrix4x3_col& _mtx)
		{
			__m256 res;
			res = _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(0)), _mm256_broadcast_ps(&_mtx.col[0]));
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(1)), _mm256_broadcast_ps(&_mtx.col[1])));
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(2)), _mm256_broadcast_ps(&_mtx.col[2])));
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(3)), _mm256_broadcast_ps(&_mtx.col[3])));
			return res;
		}
	}
#ifndef _IRR_WINDOWS_
	__attribute__((__aligned__(ALIGN)));
#endif
	; // matrix4x3_row
} // avx
#endif

#if !AVX
#define ALIGN 16
namespace sse3
{
#ifdef _IRR_WINDOWS_
	__declspec(align(ALIGN))
#endif
		struct matrix4x3_row
	{
		union
		{
			float m[4][4];
			__m128 row[4];
		};

		inline matrix4x3_row concatenate(const matrix4x3_row& _other)
		{
			matrix4x3_row out;

			out.row[0] = doJob(row[0], _other);
			out.row[1] = doJob(row[1], _other);
			out.row[2] = doJob(row[2], _other);
			out.row[3] = row[3]; // 0 0 0 1

			return out;
		}

		float& operator()(size_t _i, size_t _j) { return m[_i][_j]; }
		const float& operator()(size_t _i, size_t _j) const { return m[_i][_j]; }

	private:
		static inline __m128 doJob(const __m128& a, const matrix4x3_row& _mtx)
		{
			__m128 res;
			res = _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(0)), _mtx.row[0]);
			res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(1)), _mtx.row[1]));
			res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(2)), _mtx.row[2]));
			res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(3)), _mtx.row[3]));
			return res;
		}
	}
#ifndef _IRR_WINDOWS_
	__attribute__((__aligned__(ALIGN)));
#endif
	; // matrix4x3_row

#ifdef _IRR_WINDOWS_
	__declspec(align(ALIGN))
#endif
		struct matrix4x3_col
	{
		union
		{
			float m[4][4];
			__m128 col[4];
		};

		inline matrix4x3_col concatenate(const matrix4x3_col& _other)
		{
			matrix4x3_col out;

			out.col[0] = doJob(_other.col[0], 0, *this);
			out.col[1] = doJob(_other.col[1], 1, *this);
			out.col[2] = doJob(_other.col[2], 2, *this);
			out.col[3] = doJob(_other.col[3], 3, *this);

			return out;
		}

		float& operator()(size_t _i, size_t _j) { return m[_j][_i]; }
		const float& operator()(size_t _i, size_t _j) const { return m[_j][_i]; }

	private:
		static inline __m128 doJob(__m128 a, size_t j, const matrix4x3_col& _mtx)
		{
			__m128 res;
			res = _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(0)), _mtx.col[0]);
			res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(1)), _mtx.col[1]));
			res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(2)), _mtx.col[2]));
			if (j == 3) // w of every column except last one is 0; in the last one it's 1
				res = _mm_add_ps(res, _mtx.col[3]);
			return res;
		}
	}
#ifndef _IRR_WINDOWS_
	__attribute__((__aligned__(ALIGN)));
#endif
	; // matrix4x3_col
} // sse3
#endif

namespace measure
{
	using Clock = std::chrono::high_resolution_clock;
	using TimePoint = std::chrono::time_point<Clock>;
	using Duration = std::chrono::duration<double, std::micro>;
}

template<typename T>
static double run(void*);

int main()
{
	void* data = _aligned_malloc(16*4*(size_t)EXEC_CNT, ALIGN);
	for (size_t i = 0; i < 100; ++i)
	{
#if AVX
		printf("avx  row: %f\n", run<avx::matrix4x3_row>(data));
		printf("avx  col: %f\n", run<avx::matrix4x3_col>(data));
#else
		printf("sse3 row: %f\n", run<sse3::matrix4x3_row>(data));
		printf("sse3 col: %f\n", run<sse3::matrix4x3_col>(data));
#endif
	}
	_aligned_free(data);

	return 0;
}

template<typename T>
static double run(void* _data)
{
	T* const mtx = (T*)_data;
	for (uint32_t* p = (uint32_t*)mtx; (void*)p != (void*)(mtx+(size_t)EXEC_CNT); ++p)
		*p = rand();

	const measure::TimePoint start = measure::Clock::now();

	T m;
	for (size_t i = 0; i < (size_t)EXEC_CNT-1; ++i)
		m = mtx[i].concatenate(mtx[i+1]);

	const measure::Duration dt = measure::Clock::now() - start;
	return dt.count();
}
