#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <intrin.h>
#include <immintrin.h>

#include <chrono>
#include <random>
#include <cstdlib>
#include <cstring>
#include <limits>

// _mm256_extractf128_ps for extracting hi part
// _mm256_castps256_ps128 for extracting lo part (no cycles!)

#define EXEC_CNT (1e6)
#define BROADCAST32(fpx) _MM_SHUFFLE(fpx, fpx, fpx, fpx)

#define AVX 1 // set to 0 or 1 (sse3/avx), set appropriate compiler flags and run
#define COL_MAJOR 1 // set to 0 or 1 (col-major/row-major)
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
			__m256 row[2];
		};

		matrix4x3_row(const float* _data = 0)
		{
			if (!_data)
				return;
			memcpy(m, _data, 16*4);
		}

		inline matrix4x3_row concatenate(const matrix4x3_row& _other)
		{
			_mm256_zeroupper();
			__m256 A01 = _mm256_load_ps(&m[0][0]);
			__m256 A23 = _mm256_load_ps(&m[2][0]);

			matrix4x3_row out;

			_mm256_store_ps(&out.m[0][0], doJob(A01, _other));
			_mm256_store_ps(&out.m[2][0], doJob(A23, _other));//do special AVX 128bit ops to only load, calculate and store only one row

			return out;
		}

		float& operator()(size_t _i, size_t _j) { return m[_i][_j]; }
		const float& operator()(size_t _i, size_t _j) const { return m[_i][_j]; }

	private:
		static inline __m256 doJob(__m256 _A01, const matrix4x3_row& _mtx)
		{
			__m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, 0xffffffff, 0, 0, 0, 0xffffffff));

			__m256 res;
			res = _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(0)), _mm256_broadcast_ps(&_mtx.row[0]));
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(1)), _mm256_broadcast_ps(&_mtx.row[1])));
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(2)), _mm256_broadcast_ps(&_mtx.row[2])));
			res = _mm256_add_ps(res, _mm256_and_ps(_A01,mask));
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
			__m256 col[2];
		};

		matrix4x3_col(const float* _data = 0)
		{
			if (!_data)
				return;
			//printf("avx col ctor:\n");
			//for (size_t i = 0; i < 16; ++i)
			//	printf("%f ", _data[i]);
			//printf("\n");
			memcpy(m, _data, 16 * 4);
		}

		inline matrix4x3_col concatenate(const matrix4x3_col& _other)
		{
			__m256 A01 = _mm256_load_ps(&_other.m[0][0]);
			__m256 A23 = _mm256_load_ps(&_other.m[2][0]);

			matrix4x3_col out;

			_mm256_store_ps(&out.m[0][0], doJob(A01, 0, *this));
			_mm256_store_ps(&out.m[2][0], doJob(A23, 2, *this));

			return out;
		}

		float& operator()(size_t _i, size_t _j) { return m[_j][_i]; }
		const float& operator()(size_t _i, size_t _j) const { return m[_j][_i]; }

	private:
		static inline __m256 doJob(__m256 _A01, size_t j, const matrix4x3_col& _mtx)
		{
			__m256 res;
			res = _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(0)), _mm256_broadcast_ps(&_mtx.col[0]));
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(1)), _mm256_broadcast_ps(&_mtx.col[1])));
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_shuffle_ps(_A01, _A01, BROADCAST32(2)), _mm256_broadcast_ps(&_mtx.col[2])));
			if (j)
            {
                __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff));
                res = _mm256_add_ps(res, _mm256_and_ps(mask, _mm256_broadcast_ps(&_mtx.col[3])));
            }
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

		matrix4x3_row(const float* _data = 0)
		{
			if (!_data)
				return;
			memcpy(m, _data, 16 * 4);
		}

		inline matrix4x3_row concatenate(const matrix4x3_row& _other)
		{
			matrix4x3_row out;

			__m128 r0 = _mm_load_ps(m[0]);
			__m128 r1 = _mm_load_ps(m[1]);
			__m128 r2 = _mm_load_ps(m[2]);
			__m128 r3 = _mm_load_ps(m[3]);

			_mm_store_ps(out.m[0], doJob(r0, _other));
			_mm_store_ps(out.m[1], doJob(r1, _other));
			_mm_store_ps(out.m[2], doJob(r2, _other));
			_mm_store_ps(out.m[3], r3); // 0 0 0 1

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
			res = _mm_add_ps(res, _mm_and_ps(a, mask)); // always 0 0 0 a3 -- no shuffle needed
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

		matrix4x3_col(const float* _data = 0)
		{
			if (!_data)
				return;
			memcpy(m, _data, 16 * 4);
		}

		inline matrix4x3_col concatenate(const matrix4x3_col& _other)
		{
			matrix4x3_col out;

			__m128 c0 = _mm_load_ps(_other.m[0]);
			__m128 c1 = _mm_load_ps(_other.m[1]);
			__m128 c2 = _mm_load_ps(_other.m[2]);
			__m128 c3 = _mm_load_ps(_other.m[3]);

			_mm_store_ps(out.m[0], doJob(c0, 0, *this));
			_mm_store_ps(out.m[1], doJob(c1, 1, *this));
			_mm_store_ps(out.m[2], doJob(c2, 2, *this));
			_mm_store_ps(out.m[3], doJob(c3, 3, *this));

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

	matrix4x3_col_nosimd(const float* _data = 0)
	{
		if (!_data)
			return;
		//printf("nosimd col ctor:\n");
		//for (size_t i = 0; i < 16; ++i)
		//	printf("%f ", _data[i]);
		//printf("\n");
		memcpy(m, _data, 16*4);
	}

	inline matrix4x3_col_nosimd concatenate(const matrix4x3_col_nosimd& _other)
	{
		matrix4x3_col_nosimd ret;
		memset(ret.m, 0, 15*4);
		ret.m[3][3] = 1.f;
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

size_t cnt = 0;
int main()
{
	//const float dat1[16]{ 12, 28, 3, 0, 44, 5, 61, 0, 72, 8, 99, 0, 41, 71, 24, 1 }; // col
	//const float dat2[16]{ 22, 12, 133, 0, 44, 72, 6, 0, 17, 81, 24, 0, 14, 47, 16, 1 }; //col
	////const float dat1[16]{ 1, 2, 3, 4, 4, 5, 6, 6, 7, 8, 9, 7, 0, 0, 0, 1 }; //row
	////const float dat2[16]{ 22, 12, 133, 24, 44, 72, 6, 32, 17, 81, 24, 18, 0, 0, 0, 1 }; //row
	//avx::matrix4x3_col m1(dat1), m2(dat2);
	//avx::matrix4x3_col m3 = m1.concatenate(m2);
	//for (size_t i = 0; i < 4; ++i)
	//{
	//	for (size_t j = 0; j < 4; ++j)
	//		printf("%f ", m3.m[j][i]);
	//	printf("\n");
	//}

	//return 0; //!!!!!!!!!!

	void* data = malloc(16*4*(size_t)EXEC_CNT+ALIGN);

	uint8_t* alignedData = reinterpret_cast<uint8_t*>(data);
	size_t offset = reinterpret_cast<const size_t&>(alignedData)%ALIGN;
	alignedData += ALIGN-offset;

	{
	size_t i = 1;
	for (float* p = (float*)alignedData; (void*)p != (void*)(alignedData + (size_t)EXEC_CNT * 16 * 4); ++p, ++i)
	{
#if COL_MAJOR
		if (i % 16 == 0)
			*p = 1.f;
		else if (i % 4 == 0)
			*p = 0.f;
		else
			*p = rand();
#else
		size_t mod = i % 16;
		if (mod == 0)
			*p = 1.f;
		else if (mod >= 13 && mod <= 15)
			*p = 0.f;
		else
			*p = rand();
#endif
	}
	}

	void* dataOut = alignedData+(16*4*(size_t)EXEC_CNT);
	void* nosimdOut = alignedData+2*(16*4*(size_t)EXEC_CNT);

	double nosimdtime = 0.0;
	double simdtime = 0.0;

	for (size_t i = 0; i < 10; ++i)
		nosimdtime += run<matrix4x3_col_nosimd>(alignedData, nosimdOut, 0);

	for (size_t i = 0; i < 10; ++i)
	{
#if AVX
	#if COL_MAJOR
		simdtime += run<avx::matrix4x3_col>(alignedData, dataOut, nosimdOut);
	#else
		simdtime += run<avx::matrix4x3_row>(alignedData, dataOut, nosimdOut);
	#endif
#else
	#if COL_MAJOR
		simdtime += run<sse3::matrix4x3_col>(alignedData, dataOut, nosimdOut);
	#else
		simdtime += run<sse3::matrix4x3_row>(alignedData, dataOut, nosimdOut);
	#endif
#endif
	}

	printf("nosimd  : %f\n", nosimdtime);
#if AVX
#if COL_MAJOR
    printf("avx  col: %f\n", simdtime);
#else
	printf("avx  row: %f\n", simdtime);
#endif
#else
#if COL_MAJOR
    printf("sse3 col: %f\n", simdtime);
#else
    printf("sse3 row: %f\n", simdtime);
#endif
#endif
	free(data);
	printf("cnt: %u\n", cnt);

	return 0;
}

template<typename T>
static bool compare(T* _m1, T* _m2)
{
    const float size_thresh = 0.000001f;

	for (size_t i=0; i<4; i++)
	for (size_t j=0; j<4; j++)
    {
        const float a = (*_m1)(i,j);
        const float b = (*_m2)(i,j);
        const uint32_t& aAsInt = reinterpret_cast<const uint32_t&>(a);
        const uint32_t& bAsInt = reinterpret_cast<const uint32_t&>(b);

        if (abs(b)>size_thresh)
        {
            if (abs(1.f-abs(a/b))>0.001f)
            {
                printf("%f,%f\n",a,b);
                return false;
            }
        }
        else if ( (aAsInt&0x80000000ull)!=(bAsInt&0x80000000ull) || abs(a)>size_thresh)
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

	const float* const mtx = (float*)_data;
	T* out = (T*)_out;
	for (size_t i = 0; i < (size_t)EXEC_CNT-1; ++i)
	{
		T m1(mtx+16*i), m2(mtx+16*(i+1));
		out[i] = m1.concatenate(m2);
#if VERIFY
		if (_cmp && !compare<T>(out+i, ((T*)_cmp)+i))
		{
			printf("???\n");
			cnt++;
		}
		//if (_cmp)
		//{
		//	T& nosimd = *(((T*)_cmp) + i);
		//	printf("nosimd:\n");
		//	for (size_t i = 0; i < 4; ++i)
		//	{
		//		for (size_t j = 0; j < 4; ++j)
		//			printf("%f ", nosimd.m[j][i]);
		//		printf("\n");
		//	}
		//	T& m3 = out[i];
		//	printf("simd:\n");
		//	for (size_t i = 0; i < 4; ++i)
		//	{
		//		for (size_t j = 0; j < 4; ++j)
		//			printf("%f ", m3.m[j][i]);
		//		printf("\n");
		//	}
		//int a = 0; // breakpoint here
		//}
#endif
	}

	const measure::Duration dt = measure::Clock::now() - start;
	return dt.count();
}
