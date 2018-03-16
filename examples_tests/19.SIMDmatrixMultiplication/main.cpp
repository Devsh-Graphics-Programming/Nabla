#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <immintrin.h>

#include <chrono>
#include <random>
#include <cstdlib>
#include <cstring>

// _mm256_extractf128_ps for extracting hi part
// _mm256_castps256_ps128 for extracting lo part (no cycles!)

#define EXEC_CNT (1e6)
#define BROADCAST32(fpx) _MM_SHUFFLE(fpx, fpx, fpx, fpx)

#define AVX 1 // set to 0 or 1 (sse3/avx), set appropriate compiler flags and run
#define VERIFY 0

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

			_mm_store_ps(out.m[0], doJob(row[0], _other));
			_mm_store_ps(out.m[1], doJob(row[1], _other));
			_mm_store_ps(out.m[2], doJob(row[2], _other));
			_mm_store_ps(out.m[3], row[3]); // 0 0 0 1

			return out;
		}

		float& operator()(size_t _i, size_t _j) { return m[_i][_j]; }
		const float& operator()(size_t _i, size_t _j) const { return m[_i][_j]; }

	private:
		static inline __m128 doJob(const __m128& a, const matrix4x3_row& _mtx)
		{
			__m128 res;

			__m128 r0 = _mm_load_ps(_mtx.m[0]);
			__m128 r1 = _mm_load_ps(_mtx.m[1]);
			__m128 r2 = _mm_load_ps(_mtx.m[2]);
			__m128 r3 = _mm_load_ps(_mtx.m[3]);

			__m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0, 0, 0, 0xffffffff));

			res = _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(0)), r0);
			res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(1)), r1));
			res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(2)), r2));
			res = _mm_add_ps(res, _mm_and_ps(_mm_shuffle_ps(a, a, BROADCAST32(3)), mask)); // always 0 0 0 a3
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

			_mm_store_ps(out.m[0], doJob(_other.col[0], 0, *this));
			_mm_store_ps(out.m[1], doJob(_other.col[1], 1, *this));
			_mm_store_ps(out.m[2], doJob(_other.col[2], 2, *this));
			_mm_store_ps(out.m[3], doJob(_other.col[3], 3, *this));

			return out;
		}

		float& operator()(size_t _i, size_t _j) { return m[_j][_i]; }
		const float& operator()(size_t _i, size_t _j) const { return m[_j][_i]; }

	private:
		static inline __m128 doJob(__m128 a, size_t j, const matrix4x3_col& _mtx)
		{
			__m128 res;

			__m128 c0 = _mm_load_ps(_mtx.m[0]);
			__m128 c1 = _mm_load_ps(_mtx.m[1]);
			__m128 c2 = _mm_load_ps(_mtx.m[2]);
			__m128 c3 = _mm_load_ps(_mtx.m[3]);

			res = _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(0)), c0);
			res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(1)), c1));
			res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(2)), c2));
			if (j == 3) // w of every column except last one is 0; in the last one it's 1
				res = _mm_add_ps(res, c3);
			return res;
		}
	}
#ifndef _IRR_WINDOWS_
	__attribute__((__aligned__(ALIGN)));
#endif
	; // matrix4x3_col
} // sse3
#endif

struct matrix4x3_col_nosimd
{
	float m[4][4];

	inline matrix4x3_col_nosimd concatenate(const matrix4x3_col_nosimd& _other)
	{
		matrix4x3_col_nosimd ret;
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				float sum = 0;
				for (int e = 0; e < 3; e++)
					sum += (*this)(i, e) * _other(e, j);
				if (j == 3)
					sum += (*this)(i, 3);
				ret(i, j) = sum;
			}
		}
		return ret;
	}

	float& operator()(size_t _i, size_t _j) { return m[_j][_i]; }
	const float& operator()(size_t _i, size_t _j) const { return m[_j][_i]; }
};

namespace measure
{
	using Clock = std::chrono::high_resolution_clock;
	using TimePoint = std::chrono::time_point<Clock>;
	using Duration = std::chrono::duration<double, std::micro>;
}

template<typename T>
static bool compare(T* _m1, T* _m2);
template<typename T>
static double run(void*, void*, void*);

int main()
{
	void* data = malloc(16*4*(size_t)EXEC_CNT*3+ALIGN);

	uint8_t* alignedData = reinterpret_cast<uint8_t*>(data);
	size_t offset = reinterpret_cast<const size_t&>(alignedData)%ALIGN;
	alignedData += ALIGN-offset;

	for (uint32_t* p = (uint32_t*)alignedData; (void*)p != (void*)(alignedData + (size_t)EXEC_CNT*16*4); ++p)
    {
		*p = rand();
		*p /= 1024.f;
    }

	void* dataOut = alignedData+(16*4*(size_t)EXEC_CNT);
	void* nosimdOut = alignedData+2*(16*4*(size_t)EXEC_CNT);

	double nosimdtime = 0.0;
	double rowtime = 0.0;
	double coltime = 0.0;

	for (size_t i = 0; i < 100; ++i)
		nosimdtime += run<matrix4x3_col_nosimd>(alignedData, nosimdOut, 0);

	for (size_t i = 0; i < 100; ++i)
	{
#if AVX
		rowtime += run<avx::matrix4x3_row>(alignedData, dataOut, nosimdOut);
		coltime += run<avx::matrix4x3_col>(alignedData, dataOut, nosimdOut);
#else
		rowtime += run<sse3::matrix4x3_row>(alignedData, dataOut, nosimdOut);
		coltime += run<sse3::matrix4x3_col>(alignedData, dataOut, nosimdOut);
#endif
	}

	printf("nosimd  : %f\n", nosimdtime);
#if AVX
	printf("avx  row: %f\n", rowtime);
    printf("avx  col: %f\n", coltime);
#else
    printf("sse3 row: %f\n", rowtime);
    printf("sse3 col: %f\n", coltime);
#endif
	free(data);

	return 0;
}

template<typename T>
static bool compare(T* _m1, T* _m2) // naive cmp function for now
{
	for (size_t i=0; i<4; i++)
	for (size_t j=0; j<4; j++)
    {
        const float a = (*_m1)(i,j);
        const float b = (*_m2)(i,j);
        const uint32_t& aAsInt = reinterpret_cast<const uint32_t&>(a);
        const uint32_t& bAsInt = reinterpret_cast<const uint32_t&>(b);

        if (abs(b)>0.000001f)
        {
            if (abs(1.f-abs(a/b))<0.999f)
            {
                printf("%f,%f\n",a,b);
                return false;
            }
        }
        else if ( (aAsInt&0x80000000ull)!=(bAsInt&0x80000000ull) || abs(a)>0.000000001f)
        {
            printf("ZERO %f,%f\n",a,b);
            return false;
        }
    }
	return true;
}

template<typename T>
static double run(void* _data, void* _out, void* _cmp)
{
	const measure::TimePoint start = measure::Clock::now();

	T* const mtx = (T*)_data;
	T* out = (T*)_out;
	for (size_t i = 0; i < (size_t)EXEC_CNT-1; ++i)
	{
		out[i] = mtx[i].concatenate(mtx[i+1]);
#if VERIFY
		if (_cmp && !compare<T>(out+i, ((T*)_cmp)+i))
			printf("???\n");
#endif
	}

	const measure::Duration dt = measure::Clock::now() - start;
	return dt.count();
}
