#ifndef __MATRIX3X4SIMD_H_INCLUDED__
#define __MATRIX3X4SIMD_H_INCLUDED__

#include <quaternion.h>

namespace irr { namespace core
{
#ifdef _IRR_WINDOWS_
__declspec(align(SIMD_ALIGNMENT))
#endif
//! Equivalent of GLSL's mat4x3
struct matrix3x4SIMD
{
	core::vectorSIMDf rows[3];

	matrix3x4SIMD(const float* _data = 0)
	{
		if (!_data)
			return;
		for (size_t i = 0u; i < 3u; ++i)
			rows[i] = vectorSIMDf(_data + 4*i);
	}

	inline matrix3x4SIMD concatenate(const matrix3x4SIMD& _other)
	{
		__m128 r0 = rows[0].getAsRegister();
		__m128 r1 = rows[1].getAsRegister();
		__m128 r2 = rows[2].getAsRegister();

		matrix3x4SIMD out;
		_mm_store_ps(out.rows[0].pointer, doJob(r0, _other));
		_mm_store_ps(out.rows[1].pointer, doJob(r1, _other));
		_mm_store_ps(out.rows[2].pointer, doJob(r2, _other));

		return out;
	}

	bool operator==(const matrix3x4SIMD& _other)
	{
		for (size_t i = 0u; i < 3u; ++i)
			if (!((rows[i] == _other.rows[i]).all()))
				return false;
		return true;
	}

	bool operator!=(const matrix3x4SIMD& _other)
	{
		return !(*this == _other);
	}

	matrix3x4SIMD& operator+=(const matrix3x4SIMD& _other)
	{
		for (size_t i = 0u; i < 3u; ++i)
			rows[i] += _other.rows[i];
		return *this;
	}
	matrix3x4SIMD operator+(const matrix3x4SIMD& _other) const
	{
		matrix3x4SIMD retval(*this);
		return retval += _other;
	}

	matrix3x4SIMD& operator-=(const matrix3x4SIMD& _other)
	{
		for (size_t i = 0u; i < 3u; ++i)
			rows[i] -= _other.rows[i];
		return *this;
	}
	matrix3x4SIMD operator-(const matrix3x4SIMD& _other) const
	{
		matrix3x4SIMD retval(*this);
		return retval -= _other;
	}

	matrix3x4SIMD& operator*=(float _scalar)
	{
		for (size_t i = 0u; i < 3u; ++i)
			rows[i] *= _scalar;
		return *this;
	}
	matrix3x4SIMD operator*(float _scalar) const
	{
		matrix3x4SIMD retval(*this);
		return retval *= _scalar;
	}

	matrix3x4SIMD& setTranslation(const core::vectorSIMDf& _translation)
	{
		rows[0].w = _translation.x;
		rows[1].w = _translation.y;
		rows[2].w = _translation.z;
		return *this;
	}
	matrix3x4SIMD& setInverseTranslation(const core::vectorSIMDf& _translation)
	{
		return setTranslation(-_translation);
	}
	core::vectorSIMDf getTranslation() const
	{
		core::vectorSIMDf retval(rows[0].w, rows[1].w, rows[2].w, 1.f);
		return retval;
	}

	matrix3x4SIMD& setScale(const core::vectorSIMDf& _scale)
	{
		rows[0].x = _scale.X;
		rows[1].y = _scale.Y;
		rows[2].z = _scale.Z;

		return *this;
	}
	matrix3x4SIMD& setScale(const float _scale) { return setScale(core::vectorSIMDf(_scale, _scale, _scale)); }

	core::vectorSIMDf getScale() const
	{
		core::vectorSIMDf mask = _mm_castsi128_ps(_mm_setr_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0));
		return core::vectorSIMDf(
			(rows[0] & mask).getLengthAsFloat(),
			(rows[1] & mask).getLengthAsFloat(),
			(rows[2] & mask).getLengthAsFloat(),
			1.f
		);
	}

	inline matrix3x4SIMD& buildCameraLookAtMatrixLH(
		const core::vectorSIMDf& position,
		const core::vectorSIMDf& target,
		const core::vectorSIMDf& upVector)
	{
		const core::vectorSIMDf zaxis = core::normalize(target - position);
		const core::vectorSIMDf xaxis = core::normalize(upVector.crossProduct(zaxis));
		const core::vectorSIMDf yaxis = zaxis.crossProduct(xaxis);

		matrix3x4SIMD r;
		r.rows[0] = xaxis;
		r.rows[1] = yaxis;
		r.rows[2] = zaxis;
		r.rows[0].w = -xaxis.dotProductAsFloat(position);
		r.rows[1].w = -yaxis.dotProductAsFloat(position);
		r.rows[2].w = -zaxis.dotProductAsFloat(position);

		return r;
	}
	inline matrix3x4SIMD& buildCameraLookAtMatrixRH(
		const core::vectorSIMDf& position,
		const core::vectorSIMDf& target,
		const core::vectorSIMDf& upVector)
	{
		const core::vectorSIMDf zaxis = core::normalize(position - target);
		const core::vectorSIMDf xaxis = core::normalize(upVector.crossProduct(zaxis));
		const core::vectorSIMDf yaxis = zaxis.crossProduct(xaxis);

		matrix3x4SIMD r;
		r.rows[0] = xaxis;
		r.rows[1] = yaxis;
		r.rows[2] = zaxis;
		r.rows[0].w = -xaxis.dotProductAsFloat(position);
		r.rows[1].w = -yaxis.dotProductAsFloat(position);
		r.rows[2].w = -zaxis.dotProductAsFloat(position);

		return r;
	}

	friend inline matrix3x4SIMD mix(const matrix3x4SIMD& _a, const matrix3x4SIMD& _b, const float& _x)
	{
		matrix3x4SIMD mat;

		__m128 x = _mm_set1_ps(_x);
		__m128 ar0 = _mm_load_ps(_a.rows[0]);
		__m128 ar1 = _mm_load_ps(_a.rows[1]);
		__m128 ar2 = _mm_load_ps(_a.rows[2]);
		_mm_store_ps(mat.rows[0], _mm_add_ps(ar0, _mm_mul_ps(_mm_sub_ps(_mm_load_ps(_b.rows[0]), ar0), x)));
		_mm_store_ps(mat.rows[1], _mm_add_ps(ar1, _mm_mul_ps(_mm_sub_ps(_mm_load_ps(_b.rows[1]), ar1), x)));
		_mm_store_ps(mat.rows[2], _mm_add_ps(ar2, _mm_mul_ps(_mm_sub_ps(_mm_load_ps(_b.rows[2]), ar2), x)));

		return mat;
	}

	inline matrix3x4SIMD& setRotation(const core::quaternion& _quat)
	{
		__m128 mask0001 = _mm_castsi128_ps(_mm_setr_epi32(0, 0, 0, 0xffffffff));

		{
			__m128 stg1 = _mm_set1_ps(_quat.Y);
			stg1 = _mm_mul_ps(stg1, _mm_setr_ps(_quat.Y, _quat.X, _quat.W, 0.f));
			stg1 = _mm_mul_ps(stg1, _mm_set1_ps(2.f));
			__m128 stg2 = _mm_set1_ps(_quat.Z);
			stg2 = _mm_mul_ps(stg2, _mm_setr_ps(_quat.Z, _quat.W, _quat.X, 0.f));
			stg2 = _mm_mul_ps(stg2, _mm_setr_ps(2.f, -2.f, 2.f, 0.f));

			_mm_store_ps(rows[0].pointer, _mm_or_ps(_mm_add_ps(stg1, stg2), _mm_and_ps(rows[0].getAsRegister(), mask0001)));
			rows[0].x = 1.f - rows[0].x;
		}
		{
			__m128 stg1 = _mm_set1_ps(_quat.Z);
			stg1 = _mm_mul_ps(stg1, _mm_setr_ps(_quat.W, _quat.Z, _quat.Y, 0.f));
			stg1 = _mm_mul_ps(stg1, _mm_set1_ps(2.f));
			__m128 stg2 = _mm_set1_ps(_quat.X);
			stg2 = _mm_mul_ps(stg2, _mm_setr_ps(_quat.Y, _quat.X, _quat.W, 0.f));
			stg2 = _mm_mul_ps(stg2, _mm_setr_ps(2.f, 2.f, -2.f, 0.f));

			_mm_store_ps(rows[1].pointer, _mm_or_ps(_mm_add_ps(stg1, stg2), _mm_and_ps(rows[1].getAsRegister(), mask0001)));
			rows[1].y = 1.f - rows[1].y;
		}
		{
			__m128 stg1 = _mm_set1_ps(_quat.X);
			stg1 = _mm_mul_ps(stg1, _mm_setr_ps(_quat.Z, _quat.W, _quat.X, 0.f));
			stg1 = _mm_mul_ps(stg1, _mm_set1_ps(2.f));
			__m128 stg2 = _mm_set1_ps(_quat.Y);
			stg2 = _mm_mul_ps(stg2, _mm_setr_ps(_quat.W, _quat.Z, _quat.Y, 0.f));
			stg2 = _mm_mul_ps(stg2, _mm_setr_ps(-2.f, 2.f, 2.f, 0.f));

			_mm_store_ps(rows[2].pointer, _mm_or_ps(_mm_add_ps(stg1, stg2), _mm_and_ps(rows[2].getAsRegister(), mask0001)));
			rows[2].z = 1.f - rows[2].z;
		}

		return *this;
	}

	// see https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html
	void getInverse(matrix3x4SIMD& _out)
	{
		// transpose 3x3, we know m03 = m13 = m23 = 0
		__m128 t0 = _mm_movelh_ps(rows[0].getAsRegister(), rows[1].getAsRegister()); // 00, 01, 10, 11
		__m128 t1 = _mm_movelh_ps(rows[1].getAsRegister(), rows[0].getAsRegister()); // 02, 03, 12, 13
		_out.rows[0] = _mm_shuffle_ps(t0, rows[2].getAsRegister(), _MM_SHUFFLE(0, 2, 0, 3));
		_out.rows[1] = _mm_shuffle_ps(t0, rows[2].getAsRegister(), _MM_SHUFFLE(1, 3, 1, 3));
		_out.rows[2] = _mm_shuffle_ps(t1, rows[2].getAsRegister(), _MM_SHUFFLE(0, 2, 2, 3));

		// (SizeSqr(rows[0]), SizeSqr(rows[1]), SizeSqr(rows[2]), 0)
		__m128 sizeSqr;
		sizeSqr = _mm_mul_ps(_out.rows[0].getAsRegister(), _out.rows[0].getAsRegister());
		sizeSqr = _mm_add_ps(sizeSqr, _mm_mul_ps(_out.rows[1].getAsRegister(), _out.rows[1].getAsRegister()));
		sizeSqr = _mm_add_ps(sizeSqr, _mm_mul_ps(_out.rows[2].getAsRegister(), _out.rows[2].getAsRegister()));

		// optional test to avoid divide by 0
		__m128 one = _mm_set1_ps(1.f);
		// for each component, if(sizeSqr < SMALL_NUMBER) sizeSqr = 1;
		__m128 rSizeSqr = _mm_blendv_ps(
			_mm_div_ps(one, sizeSqr),
			one,
			_mm_cmplt_ps(sizeSqr, _mm_set1_ps(1.e-8f))
		);

		_out.rows[0] = _mm_mul_ps(_out.rows[0].getAsRegister(), rSizeSqr);
		_out.rows[1] = _mm_mul_ps(_out.rows[1].getAsRegister(), rSizeSqr);
		_out.rows[2] = _mm_mul_ps(_out.rows[2].getAsRegister(), rSizeSqr);
	}

	float& operator()(size_t _i, size_t _j) { return rows[_i].pointer[_j]; }
	const float& operator()(size_t _i, size_t _j) const { return rows[_i].pointer[_j]; }

private:
#define BROADCAST32(fpx) _MM_SHUFFLE(fpx, fpx, fpx, fpx)
	static inline __m128 doJob(const __m128& a, const matrix3x4SIMD& _mtx)
	{
		__m128 r0 = _mtx.rows[0].getAsRegister();
		__m128 r1 = _mtx.rows[1].getAsRegister();
		__m128 r2 = _mtx.rows[2].getAsRegister();

		__m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0, 0, 0, 0xffffffff));

		__m128 res;
		res = _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(0)), r0);
		res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(1)), r1));
		res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(2)), r2));
		res = _mm_add_ps(res, _mm_and_ps(a, mask)); // always 0 0 0 a3 -- no shuffle needed
		return res;
	}
#undef BROADCAST32
}
#ifndef _IRR_WINDOWS_
__attribute__((__aligned__(SIMD_ALIGNMENT)));
#endif
;
}}

#endif