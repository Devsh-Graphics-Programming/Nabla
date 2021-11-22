// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_MATRIX3X4SIMD_IMPL_H_INCLUDED_
#define _NBL_MATRIX3X4SIMD_IMPL_H_INCLUDED_

#include "matrix3x4SIMD.h"
#include "nbl/core/math/glslFunctions.tcc"

#include "matrix4x3.h"

namespace nbl::core
{





// TODO: move to another implementation header
inline quaternion::quaternion(const matrix3x4SIMD& m)
{
	const vectorSIMDf one(1.f);
	auto Qx  = m.rows[0].xxxx()^vectorSIMDu32(0,0,0x80000000u,0x80000000u);
	auto Qy  = m.rows[1].yyyy()^vectorSIMDu32(0,0x80000000u,0,0x80000000u);
	auto Qz  = m.rows[2].zzzz()^vectorSIMDu32(0,0x80000000u,0x80000000u,0);

	auto tmp = one+Qx+Qy+Qz;
	auto invscales = inversesqrt(tmp)*0.5f;
	auto scales = tmp*invscales*0.5f;

	// TODO: speed this up
	if (tmp.x > 0.0f)
	{
		X = (m(2, 1) - m(1, 2)) * invscales.x;
		Y = (m(0, 2) - m(2, 0)) * invscales.x;
		Z = (m(1, 0) - m(0, 1)) * invscales.x;
		W = scales.x;
	}
	else
	{
		if (tmp.y>0.f)
		{
			X = scales.y;
			Y = (m(0, 1) + m(1, 0)) * invscales.y;
			Z = (m(2, 0) + m(0, 2)) * invscales.y;
			W = (m(2, 1) - m(1, 2)) * invscales.y;
		}
		else if (tmp.z>0.f)
		{
			X = (m(0, 1) + m(1, 0)) * invscales.z;
			Y = scales.z;
			Z = (m(1, 2) + m(2, 1)) * invscales.z;
			W = (m(0, 2) - m(2, 0)) * invscales.z;
		}
		else
		{
			X = (m(0, 2) + m(2, 0)) * invscales.w;
			Y = (m(1, 2) + m(2, 1)) * invscales.w;
			Z = scales.w;
			W = (m(1, 0) - m(0, 1)) * invscales.w;
		}
	}

	*this = normalize(*this);
}







inline matrix3x4SIMD& matrix3x4SIMD::set(const matrix4x3& _retarded)
{
	core::matrix4SIMD c;
	for (size_t i = 0u; i < VectorCount; ++i)
		c.rows[i] = vectorSIMDf(&_retarded.getColumn(i).X);
	*this = core::transpose(c).extractSub3x4();
	for (size_t i = 0u; i < VectorCount; ++i)
		rows[i][3] = (&_retarded.getColumn(3).X)[i];

	return *this;
}
inline matrix4x3 matrix3x4SIMD::getAsRetardedIrrlichtMatrix() const
{
	auto c = core::transpose(core::matrix4SIMD(*this));

	matrix4x3 ret;
	for (size_t i = 0u; i < VectorCount; ++i)
		_mm_storeu_ps(&ret.getColumn(i).X, c.rows[i].getAsRegister());
	std::copy(c.rows[VectorCount].pointer, c.rows[VectorCount].pointer + VectorCount, &ret.getColumn(VectorCount).X);

	return ret;
}

inline bool matrix3x4SIMD::operator!=(const matrix3x4SIMD& _other)
{
	for (size_t i = 0u; i < VectorCount; ++i)
		if ((rows[i] != _other.rows[i]).any())
			return true;
	return false;
}

inline matrix3x4SIMD& matrix3x4SIMD::operator+=(const matrix3x4SIMD& _other)
{
	for (size_t i = 0u; i < VectorCount; ++i)
		rows[i] += _other.rows[i];
	return *this;
}
inline matrix3x4SIMD& matrix3x4SIMD::operator-=(const matrix3x4SIMD& _other)
{
	for (size_t i = 0u; i < VectorCount; ++i)
		rows[i] -= _other.rows[i];
	return *this;
}
inline matrix3x4SIMD& matrix3x4SIMD::operator*=(float _scalar)
{
	for (size_t i = 0u; i < VectorCount; ++i)
		rows[i] *= _scalar;
	return *this;
}

#ifdef __NBL_COMPILE_WITH_SSE3
#define BROADCAST32(fpx) _MM_SHUFFLE(fpx, fpx, fpx, fpx)
#define BUILD_XORMASKF(_x_, _y_, _z_, _w_) _mm_setr_epi32(_x_ ? 0x80000000u:0x0u, _y_ ? 0x80000000u:0x0u, _z_ ? 0x80000000u:0x0u, _w_ ? 0x80000000u:0x0u)
#define BUILD_MASKF(_x_, _y_, _z_, _w_) _mm_setr_epi32(_x_*0xffffffff, _y_*0xffffffff, _z_*0xffffffff, _w_*0xffffffff)

inline matrix3x4SIMD matrix3x4SIMD::concatenateBFollowedByA(const matrix3x4SIMD& _a, const matrix3x4SIMD& _b)
{
#ifdef _NBL_DEBUG
	assert(is_aligned_to(&_a, _NBL_SIMD_ALIGNMENT));
	assert(is_aligned_to(&_b, _NBL_SIMD_ALIGNMENT));
#endif // _NBL_DEBUG
	__m128 r0 = _a.rows[0].getAsRegister();
	__m128 r1 = _a.rows[1].getAsRegister();
	__m128 r2 = _a.rows[2].getAsRegister();

	matrix3x4SIMD out;
	out.rows[0] = matrix3x4SIMD::doJob(r0, _b);
	out.rows[1] = matrix3x4SIMD::doJob(r1, _b);
	out.rows[2] = matrix3x4SIMD::doJob(r2, _b);

	return out;
}

inline matrix3x4SIMD matrix3x4SIMD::concatenateBFollowedByAPrecisely(const matrix3x4SIMD& _a, const matrix3x4SIMD& _b)
{
	__m128d r00 = _a.halfRowAsDouble(0u, true);
	__m128d r01 = _a.halfRowAsDouble(0u, false);
	__m128d r10 = _a.halfRowAsDouble(1u, true);
	__m128d r11 = _a.halfRowAsDouble(1u, false);
	__m128d r20 = _a.halfRowAsDouble(2u, true);
	__m128d r21 = _a.halfRowAsDouble(2u, false);

	matrix3x4SIMD out;

	const __m128i mask0011 = BUILD_MASKF(0, 0, 1, 1);

	__m128 second = _mm_cvtpd_ps(matrix3x4SIMD::doJob_d(r00, r01, _b, false));
	out.rows[0] = vectorSIMDf(_mm_cvtpd_ps(matrix3x4SIMD::doJob_d(r00, r01, _b, true))) | _mm_castps_si128((vectorSIMDf(_mm_movelh_ps(second, second)) & mask0011).getAsRegister());

	second = _mm_cvtpd_ps(matrix3x4SIMD::doJob_d(r10, r11, _b, false));
	out.rows[1] = vectorSIMDf(_mm_cvtpd_ps(matrix3x4SIMD::doJob_d(r10, r11, _b, true))) | _mm_castps_si128((vectorSIMDf(_mm_movelh_ps(second, second)) & mask0011).getAsRegister());

	second = _mm_cvtpd_ps(matrix3x4SIMD::doJob_d(r20, r21, _b, false));
	out.rows[2] = vectorSIMDf(_mm_cvtpd_ps(matrix3x4SIMD::doJob_d(r20, r21, _b, true))) | _mm_castps_si128((vectorSIMDf(_mm_movelh_ps(second, second)) & mask0011).getAsRegister());

	return out;
}

inline vectorSIMDf matrix3x4SIMD::getTranslation() const
{
	__m128 xmm0 = _mm_unpackhi_ps(rows[0].getAsRegister(), rows[1].getAsRegister()); // (0z,1z,0w,1w)
	__m128 xmm1 = _mm_unpackhi_ps(rows[2].getAsRegister(), _mm_setr_ps(0.f, 0.f, 0.f, 1.f)); // (2z,3z,2w,3w)
	__m128 xmm2 = _mm_movehl_ps(xmm1, xmm0);// (0w,1w,2w,3w)

	return xmm2;
}
inline vectorSIMDf matrix3x4SIMD::getTranslation3D() const
{
	__m128 xmm0 = _mm_unpackhi_ps(rows[0].getAsRegister(), rows[1].getAsRegister()); // (0z,1z,0w,1w)
	__m128 xmm1 = _mm_unpackhi_ps(rows[2].getAsRegister(), _mm_setzero_ps()); // (2z,0,2w,0)
	__m128 xmm2 = _mm_movehl_ps(xmm1, xmm0);// (0w,1w,2w,0)

	return xmm2;
}

inline matrix3x4SIMD& matrix3x4SIMD::setScale(const core::vectorSIMDf& _scale)
{
	const vectorSIMDu32 mask0001 = vectorSIMDu32(BUILD_MASKF(0, 0, 0, 1));
	const vectorSIMDu32 mask0010 = vectorSIMDu32(BUILD_MASKF(0, 0, 1, 0));
	const vectorSIMDu32 mask0100 = vectorSIMDu32(BUILD_MASKF(0, 1, 0, 0));
	const vectorSIMDu32 mask1000 = vectorSIMDu32(BUILD_MASKF(1, 0, 0, 0));

	const vectorSIMDu32& scaleAlias = reinterpret_cast<const vectorSIMDu32&>(_scale);

	vectorSIMDu32& rowAlias0 = reinterpret_cast<vectorSIMDu32&>(rows[0]);
	vectorSIMDu32& rowAlias1 = reinterpret_cast<vectorSIMDu32&>(rows[1]);
	vectorSIMDu32& rowAlias2 = reinterpret_cast<vectorSIMDu32&>(rows[2]);
	rowAlias0 = (scaleAlias & reinterpret_cast<const vectorSIMDf&>(mask1000)) | (rowAlias0 & reinterpret_cast<const vectorSIMDf&>(mask0001));
	rowAlias1 = (scaleAlias & reinterpret_cast<const vectorSIMDf&>(mask0100)) | (rowAlias1 & reinterpret_cast<const vectorSIMDf&>(mask0001));
	rowAlias2 = (scaleAlias & reinterpret_cast<const vectorSIMDf&>(mask0010)) | (rowAlias2 & reinterpret_cast<const vectorSIMDf&>(mask0001));

	return *this;
}

inline core::vectorSIMDf matrix3x4SIMD::getScale() const
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

inline void matrix3x4SIMD::transformVect(vectorSIMDf& _out, const vectorSIMDf& _in) const
{
	vectorSIMDf r0 = rows[0] * _in,
		r1 = rows[1] * _in,
		r2 = rows[2] * _in;

	_out =
		_mm_hadd_ps(
			_mm_hadd_ps(r0.getAsRegister(), r1.getAsRegister()),
			_mm_hadd_ps(r2.getAsRegister(), _mm_set1_ps(0.25f))
		);
}

inline void matrix3x4SIMD::pseudoMulWith4x1(vectorSIMDf& _out, const vectorSIMDf& _in) const
{
	__m128i mask1110 = BUILD_MASKF(1, 1, 1, 0);
	_out = (_in & mask1110) | _mm_castps_si128(vectorSIMDf(0.f, 0.f, 0.f, 1.f).getAsRegister());
	transformVect(_out);
}

inline void matrix3x4SIMD::mulSub3x3WithNx1(vectorSIMDf& _out, const vectorSIMDf& _in) const
{
	auto maskedIn = _in & BUILD_MASKF(1, 1, 1, 0);
	vectorSIMDf r0 = rows[0] * maskedIn,
		r1 = rows[1] * maskedIn,
		r2 = rows[2] * maskedIn;

	_out =
		_mm_hadd_ps(
			_mm_hadd_ps(r0.getAsRegister(), r1.getAsRegister()),
			_mm_hadd_ps(r2.getAsRegister(), _mm_setzero_ps())
		);
}


inline matrix3x4SIMD matrix3x4SIMD::buildCameraLookAtMatrixLH(
	const core::vectorSIMDf& position,
	const core::vectorSIMDf& target,
	const core::vectorSIMDf& upVector)
{
	const core::vectorSIMDf zaxis = core::normalize(target - position);
	const core::vectorSIMDf xaxis = core::normalize(core::cross(upVector, zaxis));
	const core::vectorSIMDf yaxis = core::cross(zaxis, xaxis);

	matrix3x4SIMD r;
	r.rows[0] = xaxis;
	r.rows[1] = yaxis;
	r.rows[2] = zaxis;
	r.rows[0].w = -dot(xaxis, position)[0];
	r.rows[1].w = -dot(yaxis, position)[0];
	r.rows[2].w = -dot(zaxis, position)[0];

	return r;
}
inline matrix3x4SIMD matrix3x4SIMD::buildCameraLookAtMatrixRH(
	const core::vectorSIMDf& position,
	const core::vectorSIMDf& target,
	const core::vectorSIMDf& upVector)
{
	const core::vectorSIMDf zaxis = core::normalize(position - target);
	const core::vectorSIMDf xaxis = core::normalize(core::cross(upVector, zaxis));
	const core::vectorSIMDf yaxis = core::cross(zaxis, xaxis);

	matrix3x4SIMD r;
	r.rows[0] = xaxis;
	r.rows[1] = yaxis;
	r.rows[2] = zaxis;
	r.rows[0].w = -dot(xaxis, position)[0];
	r.rows[1].w = -dot(yaxis, position)[0];
	r.rows[2].w = -dot(zaxis, position)[0];

	return r;
}

inline matrix3x4SIMD& matrix3x4SIMD::setRotation(const core::quaternion& _quat)
{
	const vectorSIMDu32 mask0001 = vectorSIMDu32(BUILD_MASKF(0, 0, 0, 1));
	const __m128i mask1110 = BUILD_MASKF(1, 1, 1, 0);

	const core::vectorSIMDf& quat = reinterpret_cast<const core::vectorSIMDf&>(_quat);
	rows[0] = ((quat.yyyy() * ((quat.yxwx() & mask1110) * vectorSIMDf(2.f))) + (quat.zzzz() * (quat.zwxx() & mask1110) * vectorSIMDf(2.f, -2.f, 2.f, 0.f))) | (reinterpret_cast<const vectorSIMDu32&>(rows[0]) & (mask0001));
	rows[0].x = 1.f - rows[0].x;

	rows[1] = ((quat.zzzz() * ((quat.wzyx() & mask1110) * vectorSIMDf(2.f))) + (quat.xxxx() * (quat.yxwx() & mask1110) * vectorSIMDf(2.f, 2.f, -2.f, 0.f))) | (reinterpret_cast<const vectorSIMDu32&>(rows[1]) & (mask0001));
	rows[1].y = 1.f - rows[1].y;

	rows[2] = ((quat.xxxx() * ((quat.zwxx() & mask1110) * vectorSIMDf(2.f))) + (quat.yyyy() * (quat.wzyx() & mask1110) * vectorSIMDf(-2.f, 2.f, 2.f, 0.f))) | (reinterpret_cast<const vectorSIMDu32&>(rows[2]) & (mask0001));
	rows[2].z = 1.f - rows[2].z;

	return *this;
}

inline matrix3x4SIMD& matrix3x4SIMD::setScaleRotationAndTranslation(const vectorSIMDf& _scale, const core::quaternion& _quat, const vectorSIMDf& _translation)
{
	const __m128i mask1110 = BUILD_MASKF(1, 1, 1, 0);

	const vectorSIMDf& quat = reinterpret_cast<const vectorSIMDf&>(_quat);
	const vectorSIMDf dblScale = (_scale * 2.f) & mask1110;

	vectorSIMDf mlt = dblScale ^ BUILD_XORMASKF(0, 1, 0, 0);
	rows[0] = ((quat.yyyy() * ((quat.yxwx() & mask1110) * dblScale)) + (quat.zzzz() * (quat.zwxx() & mask1110) * mlt));
	rows[0].x = _scale.x - rows[0].x;

	mlt = dblScale ^ BUILD_XORMASKF(0, 0, 1, 0);
	rows[1] = ((quat.zzzz() * ((quat.wzyx() & mask1110) * dblScale)) + (quat.xxxx() * (quat.yxwx() & mask1110) * mlt));
	rows[1].y = _scale.y - rows[1].y;

	mlt = dblScale ^ BUILD_XORMASKF(1, 0, 0, 0);
	rows[2] = ((quat.xxxx() * ((quat.zwxx() & mask1110) * dblScale)) + (quat.yyyy() * (quat.wzyx() & mask1110) * mlt));
	rows[2].z = _scale.z - rows[2].z;

	setTranslation(_translation);

	return *this;
}


inline bool matrix3x4SIMD::getInverse(matrix3x4SIMD& _out) const //! SUBOPTIMAL - OPTIMIZE!
{
	auto translation = getTranslation();
	// `tmp` will have columns in its `rows`
	core::matrix4SIMD tmp;
	auto* cols = tmp.rows;
	if (!getSub3x3InverseTranspose(reinterpret_cast<core::matrix3x4SIMD&>(tmp)))
		return false;

	// find inverse post-translation
	cols[3] = -cols[0]*translation.xxxx()-cols[1]*translation.yyyy()-cols[2]*translation.zzzz();

	// columns into rows
	_out = transpose(tmp).extractSub3x4();

	return true;
}

inline bool matrix3x4SIMD::getSub3x3InverseTranspose(core::matrix3x4SIMD& _out) const
{
	vectorSIMDf r1crossr2;
	const vectorSIMDf d = determinant_helper(r1crossr2);
	if (core::iszero(d.x, FLT_MIN))
		return false;
	auto rcp = core::reciprocal(d);

	// matrix of cofactors * 1/det
	_out = getSub3x3TransposeCofactors();
	_out.rows[0] *= rcp;
	_out.rows[1] *= rcp;
	_out.rows[2] *= rcp;

	return true;
}

inline core::matrix3x4SIMD matrix3x4SIMD::getSub3x3TransposeCofactors() const
{
	core::matrix3x4SIMD _out;
	_out.rows[0] = core::cross(rows[1], rows[2]);
	_out.rows[1] = core::cross(rows[2], rows[0]);
	_out.rows[2] = core::cross(rows[0], rows[1]);
	return _out;
}

// TODO: Double check this!-
inline void matrix3x4SIMD::setTransformationCenter(const core::vectorSIMDf& _center, const core::vectorSIMDf& _translation)
{
	core::vectorSIMDf r0 = rows[0] * _center;
	core::vectorSIMDf r1 = rows[1] * _center;
	core::vectorSIMDf r2 = rows[2] * _center;
	core::vectorSIMDf r3(0.f, 0.f, 0.f, 1.f);

	__m128 col3 = _mm_hadd_ps(_mm_hadd_ps(r0.getAsRegister(), r1.getAsRegister()), _mm_hadd_ps(r2.getAsRegister(), r3.getAsRegister()));
	const vectorSIMDf vcol3 = _center - _translation - col3;

	for (size_t i = 0u; i < VectorCount; ++i)
		rows[i].w = vcol3.pointer[i];
}


// TODO: Double check this!
inline matrix3x4SIMD matrix3x4SIMD::buildAxisAlignedBillboard(
	const core::vectorSIMDf& camPos,
	const core::vectorSIMDf& center,
	const core::vectorSIMDf& translation,
	const core::vectorSIMDf& axis,
	const core::vectorSIMDf& from)
{
	// axis of rotation
	const core::vectorSIMDf up = core::normalize(axis);
	const core::vectorSIMDf forward = core::normalize(camPos - center);
	const core::vectorSIMDf right = core::normalize(core::cross(up, forward));

	// correct look vector
	const core::vectorSIMDf look = core::cross(right, up);

	// rotate from to
	// axis multiplication by sin
	const core::vectorSIMDf vs = core::cross(look, from);

	// cosinus angle
	const core::vectorSIMDf ca = core::cross(from, look);

	const core::vectorSIMDf vt(up * (core::vectorSIMDf(1.f) - ca));
	const core::vectorSIMDf wt = vt * up.yzxx();
	const core::vectorSIMDf vtuppca = vt * up + ca;

	matrix3x4SIMD mat;
	core::vectorSIMDf& row0 = mat.rows[0];
	core::vectorSIMDf& row1 = mat.rows[1];
	core::vectorSIMDf& row2 = mat.rows[2];

	row0 = vtuppca & BUILD_MASKF(1, 0, 0, 0);
	row1 = vtuppca & BUILD_MASKF(0, 1, 0, 0);
	row2 = vtuppca & BUILD_MASKF(0, 0, 1, 0);

	row0 += (wt.xxzx() + vs.xzyx() * core::vectorSIMDf(1.f, 1.f, -1.f, 1.f)) & BUILD_MASKF(0, 1, 1, 0);
	row1 += (wt.xxyx() + vs.zxxx() * core::vectorSIMDf(-1.f, 1.f, 1.f, 1.f)) & BUILD_MASKF(1, 0, 1, 0);
	row2 += (wt.zyxx() + vs.yxxx() * core::vectorSIMDf(1.f, -1.f, 1.f, 1.f)) & BUILD_MASKF(1, 1, 0, 0);

	mat.setTransformationCenter(center, translation);
	return mat;
}



inline vectorSIMDf matrix3x4SIMD::doJob(const __m128& a, const matrix3x4SIMD& _mtx)
{
	__m128 r0 = _mtx.rows[0].getAsRegister();
	__m128 r1 = _mtx.rows[1].getAsRegister();
	__m128 r2 = _mtx.rows[2].getAsRegister();

	const __m128i mask = _mm_setr_epi32(0, 0, 0, 0xffffffff);

	vectorSIMDf res;
	res = _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(0)), r0);
	res += _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(1)), r1);
	res += _mm_mul_ps(_mm_shuffle_ps(a, a, BROADCAST32(2)), r2);
	res += vectorSIMDf(a) & mask; // always 0 0 0 a3 -- no shuffle needed
	return res;
	}

inline __m128d matrix3x4SIMD::halfRowAsDouble(size_t _n, bool _0) const
{
	return _mm_cvtps_pd(_0 ? rows[_n].xyxx().getAsRegister() : rows[_n].zwxx().getAsRegister());
}
inline __m128d matrix3x4SIMD::doJob_d(const __m128d& _a0, const __m128d& _a1, const matrix3x4SIMD& _mtx, bool _xyHalf)
{
	__m128d r0 = _mtx.halfRowAsDouble(0u, _xyHalf);
	__m128d r1 = _mtx.halfRowAsDouble(1u, _xyHalf);
	__m128d r2 = _mtx.halfRowAsDouble(2u, _xyHalf);

	const __m128d mask01 = _mm_castsi128_pd(_mm_setr_epi32(0, 0, 0xffffffff, 0xffffffff));

	__m128d res;
	res = _mm_mul_pd(_mm_shuffle_pd(_a0, _a0, 0), r0);
	res = _mm_add_pd(res, _mm_mul_pd(_mm_shuffle_pd(_a0, _a0, 3), r1));
	res = _mm_add_pd(res, _mm_mul_pd(_mm_shuffle_pd(_a1, _a1, 0), r2));
	if (!_xyHalf)
		res = _mm_add_pd(res, _mm_and_pd(_a1, mask01));
	return res;
}

#undef BUILD_MASKF
#undef BUILD_XORMASKF
#undef BROADCAST32
#else
#error "no implementation"
#endif

} // nbl::core

#endif
