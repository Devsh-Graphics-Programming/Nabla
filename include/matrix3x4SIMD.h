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

	matrix3x4SIMD(const vectorSIMDf& _r0, const vectorSIMDf& _r1, const vectorSIMDf& _r2) : rows{_r0, _r1, _r2} {}

	matrix3x4SIMD(
		float _a00, float _a01, float _a02, float _a03,
		float _a10, float _a11, float _a12, float _a13,
		float _a20, float _a21, float _a22, float _a23) 
	: matrix3x4SIMD(vectorSIMDf(_a00, _a01, _a02, _a03), vectorSIMDf(_a10, _a11, _a12, _a13), vectorSIMDf(_a20, _a21, _a22, _a23))
	{
	}

	explicit matrix3x4SIMD(const float* const _data = nullptr)
	{
		if (!_data)
			return;
		for (size_t i = 0u; i < 3u; ++i)
			rows[i] = vectorSIMDf(_data + 4*i);
	}

	friend inline matrix3x4SIMD concatenateBFollowedByA(const matrix3x4SIMD& _a, const matrix3x4SIMD& _b)
	{
		__m128 r0 = _a.rows[0].getAsRegister();
		__m128 r1 = _a.rows[1].getAsRegister();
		__m128 r2 = _a.rows[2].getAsRegister();

		matrix3x4SIMD out;
		_mm_store_ps(out.rows[0].pointer, matrix3x4SIMD::doJob(r0, _b));
		_mm_store_ps(out.rows[1].pointer, matrix3x4SIMD::doJob(r1, _b));
		_mm_store_ps(out.rows[2].pointer, matrix3x4SIMD::doJob(r2, _b));

		return out;
	}

	inline matrix3x4SIMD& concatenateAfter(const matrix3x4SIMD& _other)
	{
		return *this = concatenateBFollowedByA(*this, _other);
	}

	inline matrix3x4SIMD& concatenateBefore(const matrix3x4SIMD& _other)
	{
		return *this = concatenateBFollowedByA(_other, *this);
	}

	inline bool operator==(const matrix3x4SIMD& _other)
	{
		for (size_t i = 0u; i < 3u; ++i)
			if (!((rows[i] == _other.rows[i]).all()))
				return false;
		return true;
	}

	inline bool operator!=(const matrix3x4SIMD& _other)
	{
		return !(*this == _other);
	}

	inline matrix3x4SIMD& operator+=(const matrix3x4SIMD& _other)
	{
		for (size_t i = 0u; i < 3u; ++i)
			rows[i] += _other.rows[i];
		return *this;
	}
	inline matrix3x4SIMD operator+(const matrix3x4SIMD& _other) const
	{
		matrix3x4SIMD retval(*this);
		return retval += _other;
	}

	inline matrix3x4SIMD& operator-=(const matrix3x4SIMD& _other)
	{
		for (size_t i = 0u; i < 3u; ++i)
			rows[i] -= _other.rows[i];
		return *this;
	}
	inline matrix3x4SIMD operator-(const matrix3x4SIMD& _other) const
	{
		matrix3x4SIMD retval(*this);
		return retval -= _other;
	}

	inline matrix3x4SIMD& operator*=(float _scalar)
	{
		for (size_t i = 0u; i < 3u; ++i)
			rows[i] *= _scalar;
		return *this;
	}
	inline matrix3x4SIMD operator*(float _scalar) const
	{
		matrix3x4SIMD retval(*this);
		return retval *= _scalar;
	}

	inline matrix3x4SIMD& setTranslation(const core::vectorSIMDf& _translation)
	{
		rows[0].w = _translation.x;
		rows[1].w = _translation.y;
		rows[2].w = _translation.z;
		return *this;
	}
	inline matrix3x4SIMD& setInverseTranslation(const core::vectorSIMDf& _translation)
	{
		return setTranslation(-_translation);
	}
	inline vectorSIMDf getTranslation() const
	{
		__m128 xmm0 = _mm_unpackhi_ps(rows[0].getAsRegister(), rows[1].getAsRegister()); // (0z,1z,0w,1w)
		__m128 xmm1 = _mm_unpackhi_ps(rows[2].getAsRegister(), rows[3].getAsRegister()); // (2z,3z,2w,3w)
		__m128 xmm2 = _mm_movehl_ps(xmm1, xmm0);// (0w,1w,2w,3w)

		return xmm2;
	}
	inline vectorSIMDf getTranslation3D() const
	{
		__m128 xmm0 = _mm_unpackhi_ps(rows[0].getAsRegister(), rows[1].getAsRegister()); // (0z,1z,0w,1w)
		__m128 xmm1 = _mm_unpackhi_ps(rows[2].getAsRegister(), _mm_setzero_ps()); // (2z,0,2w,0)
		__m128 xmm2 = _mm_movehl_ps(xmm1, xmm0);// (0w,1w,2w,0)

		return xmm2;
	}

	inline matrix3x4SIMD& setScale(const core::vectorSIMDf& _scale)
	{
		rows[0].x = _scale.X;
		rows[1].y = _scale.Y;
		rows[2].z = _scale.Z;

		return *this;
	}
	inline matrix3x4SIMD& setScale(const float _scale) { return setScale(core::vectorSIMDf(_scale, _scale, _scale)); }

	inline core::vectorSIMDf getScale() const
	{
		// xmm4-7 will now become columuns of B
		__m128 xmm4 = rows[0].getAsRegister();
		__m128 xmm5 = rows[1].getAsRegister();
		__m128 xmm6 = rows[2].getAsRegister();
		__m128 xmm7 = _mm_setzero_ps();
		// g==0
		__m128 xmm0 = _mm_unpacklo_ps(xmm4, xmm5);
		__m128 xmm1 = _mm_unpacklo_ps(xmm6, xmm7); // (2x,g,2y,g)
		__m128 xmm2 = _mm_unpackhi_ps(xmm4, xmm5);
		__m128 xmm3 = _mm_unpackhi_ps(xmm6, xmm7); // (2z,g,2w,g)
		xmm4 = _mm_movelh_ps(xmm1, xmm0); //(0x,1x,2x,g)
		xmm5 = _mm_movehl_ps(xmm1, xmm0);
		xmm6 = _mm_movelh_ps(xmm3, xmm2); //(0z,1z,2z,g)

		// See http://www.robertblum.com/articles/2005/02/14/decomposing-matrices
		// We have to do the full calculation.
		xmm0 = _mm_mul_ps(xmm4, xmm4);// column 0 squared
		xmm1 = _mm_mul_ps(xmm5, xmm5);// column 1 squared
		xmm2 = _mm_mul_ps(xmm6, xmm6);// column 2 squared
		xmm4 = _mm_hadd_ps(xmm0, xmm1);
		xmm5 = _mm_hadd_ps(xmm2, xmm7);
		xmm6 = _mm_hadd_ps(xmm4, xmm5);

		return _mm_sqrt_ps(xmm6);
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

		mat.rows[0] = core::mix(_a.rows[0], _b.rows[0], vectorSIMDf(_x));
		mat.rows[1] = core::mix(_a.rows[1], _b.rows[1], vectorSIMDf(_x));
		mat.rows[2] = core::mix(_a.rows[2], _b.rows[2], vectorSIMDf(_x));

		return mat;
	}

	inline matrix3x4SIMD& setRotation(const core::quaternion& _quat)
	{
		const __m128 mask0001 = _mm_castsi128_ps(_mm_setr_epi32(0, 0, 0, 0xffffffff));
		const __m128 mask1110 = _mm_castsi128_ps(_mm_setr_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0));

		rows[0] = ((_quat.yyyy() * ((_quat.yxwx() & mask1110) * vectorSIMDf(2.f))) + (_quat.zzzz() * (_quat.zwxx() & mask1110) * vectorSIMDf(2.f, -2.f, 2.f, 0.f))) | (rows[0] & mask0001);
		rows[0].x = 1.f - rows[0].x;

		rows[1] = ((_quat.zzzz() * ((_quat.wzyx() & mask1110) * vectorSIMDf(2.f))) + (_quat.xxxx() * (_quat.yxwx() & mask1110) * vectorSIMDf(2.f, 2.f, -2.f, 0.f))) | (rows[1] & mask0001);
		rows[1].y = 1.f - rows[1].y;

		rows[2] = ((_quat.xxxx() * ((_quat.zwxx() & mask1110) * vectorSIMDf(2.f))) + (_quat.yyyy() * (_quat.wzyx() & mask1110) * vectorSIMDf(-2.f, 2.f, 2.f, 0.f))) | (rows[2] & mask0001);
		rows[2].z = 1.f - rows[2].z;

		return *this;
	}

	//! @bug @todo not fine
	inline bool getInverse(matrix3x4SIMD& _out)
	{
		// assuming 4th row being 0,0,0,1 like in a sane 4x4 matrix used in games
		vectorSIMDf tmpA = rows[1].zxxw()*rows[2].yzyw();// (m(1, 2) * m(2, 1)
		vectorSIMDf tmpB = rows[1].yzyw()*rows[2].zxxw();// (m(1, 1) * m(2, 2))
		vectorSIMDf tmpC = tmpA - tmpB; //1st column of _out matrix
		vectorSIMDf preDeterminant = rows[0] * tmpC;
		preDeterminant = _mm_hadd_ps(preDeterminant.getAsRegister(), preDeterminant.getAsRegister()); // (x+y,z+w,..)
		vectorSIMDf determinant4 = _mm_hadd_ps(preDeterminant.getAsRegister(), preDeterminant.getAsRegister());

		if (determinant4.x == 0.f)
			return false;

		tmpA = rows[0].zxyw()*rows[2].yzxw();
		tmpB = rows[0].yzxw()*rows[2].zxyw();
		vectorSIMDf tmpD = tmpA - tmpB; // 2nd column of _out matrix

		tmpA = rows[0].yzxw()*rows[1].zxyw();
		tmpB = rows[0].zxyw()*rows[1].yzxw();
		vectorSIMDf tmpE = tmpA - tmpB; // 3rd column of _out matrix

		__m128 xmm0 = tmpC.getAsRegister();
		__m128 xmm1 = tmpD.getAsRegister();
		__m128 xmm2 = tmpE.getAsRegister();
		__m128 xmm3 = _mm_setzero_ps();

		_MM_TRANSPOSE4_PS(xmm0, xmm1, xmm2, xmm3);

		__m128 xmm4 = getTranslation3D().getAsRegister();

		xmm0 = _mm_mul_ps(xmm0, xmm4); //_out(0,3)
		xmm1 = _mm_mul_ps(xmm1, xmm4); //_out(1,3)
		xmm2 = _mm_or_ps(_mm_mul_ps(xmm2, xmm4), _mm_castsi128_ps(_mm_set_epi32(0, -1, 0, -1))); //_out(2,3)

		xmm0 = _mm_hsub_ps(xmm0, xmm1); // C.x-D.x,E.x,C.y-D.y,E.y
		xmm1 = _mm_hsub_ps(xmm2, preDeterminant.getAsRegister()); // C.z-D.z,E.z,x+y-z-w,x+Y-z-w
		xmm2 = _mm_hsub_ps(xmm0, xmm1); // C.x-D.x-E.x,C.y-D.y-E.y,C.z-D.z-E.z,0

		_MM_TRANSPOSE4_PS(tmpC.getAsRegister(), tmpD.getAsRegister(), tmpE.getAsRegister(), xmm2);
		_out.rows[0] = tmpC;
		_out.rows[1] = tmpD;
		_out.rows[2] = tmpE;

		__m128 rdet = _mm_rcp_ps(determinant4.getAsRegister());
		_out.rows[0] *= rdet;
		_out.rows[1] *= rdet;
		_out.rows[2] *= rdet;

		return true;
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