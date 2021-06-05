// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_MATRIX4SIMD_IMPL_H_INCLUDED__
#define __NBL_MATRIX4SIMD_IMPL_H_INCLUDED__

#include "matrix4SIMD.h"
#include "nbl/core/math/glslFunctions.tcc"

namespace nbl
{
namespace core
{


inline bool matrix4SIMD::operator!=(const matrix4SIMD& _other) const
{
	for (size_t i = 0u; i < VectorCount; ++i)
		if ((rows[i] != _other.rows[i]).any())
			return true;
	return false;
}

inline matrix4SIMD& matrix4SIMD::operator+=(const matrix4SIMD& _other)
{
	for (size_t i = 0u; i < VectorCount; ++i)
		rows[i] += _other.rows[i];
	return *this;
}

inline matrix4SIMD& matrix4SIMD::operator-=(const matrix4SIMD& _other)
{
	for (size_t i = 0u; i < VectorCount; ++i)
		rows[i] -= _other.rows[i];
	return *this;
}

inline matrix4SIMD& matrix4SIMD::operator*=(float _scalar)
{
	for (size_t i = 0u; i < VectorCount; ++i)
		rows[i] *= _scalar;
	return *this;
}

inline bool matrix4SIMD::isIdentity(float _tolerance) const
{
	return core::equals<matrix4SIMD>(*this, matrix4SIMD(), core::ROUNDING_ERROR<matrix4SIMD>());
}

#ifdef __NBL_COMPILE_WITH_SSE3
#define BROADCAST32(fpx) _MM_SHUFFLE(fpx, fpx, fpx, fpx)
#define BUILD_MASKF(_x_, _y_, _z_, _w_) _mm_setr_epi32(_x_*0xffffffff, _y_*0xffffffff, _z_*0xffffffff, _w_*0xffffffff)
inline matrix4SIMD matrix4SIMD::concatenateBFollowedByA(const matrix4SIMD& _a, const matrix4SIMD& _b)
{
	auto calcRow = [](const __m128& _row, const matrix4SIMD& _mtx)
	{
		__m128 r0 = _mtx.rows[0].getAsRegister();
		__m128 r1 = _mtx.rows[1].getAsRegister();
		__m128 r2 = _mtx.rows[2].getAsRegister();
		__m128 r3 = _mtx.rows[3].getAsRegister();

		__m128 res;
		res = _mm_mul_ps(_mm_shuffle_ps(_row, _row, BROADCAST32(0)), r0);
		res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(_row, _row, BROADCAST32(1)), r1));
		res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(_row, _row, BROADCAST32(2)), r2));
		res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(_row, _row, BROADCAST32(3)), r3));
		return res;
	};

	matrix4SIMD r;
	for (size_t i = 0u; i < 4u; ++i)
		r.rows[i] = calcRow(_a.rows[i].getAsRegister(), _b);

	return r;
}
inline matrix4SIMD matrix4SIMD::concatenateBFollowedByAPrecisely(const matrix4SIMD& _a, const matrix4SIMD& _b)
{
	matrix4SIMD out;

	__m128i mask0011 = BUILD_MASKF(0, 0, 1, 1);
	__m128 second;

	{
	__m128d r00 = _a.halfRowAsDouble(0u, true);
	__m128d r01 = _a.halfRowAsDouble(0u, false);
	second = _mm_cvtpd_ps(concat64_helper(r00, r01, _b, false));
	out.rows[0] = vectorSIMDf(_mm_cvtpd_ps(concat64_helper(r00, r01, _b, true))) | _mm_castps_si128((vectorSIMDf(_mm_movelh_ps(second, second)) & mask0011).getAsRegister());
	}

	{
	__m128d r10 = _a.halfRowAsDouble(1u, true);
	__m128d r11 = _a.halfRowAsDouble(1u, false);
	second = _mm_cvtpd_ps(concat64_helper(r10, r11, _b, false));
	out.rows[1] = vectorSIMDf(_mm_cvtpd_ps(concat64_helper(r10, r11, _b, true))) | _mm_castps_si128((vectorSIMDf(_mm_movelh_ps(second, second)) & mask0011).getAsRegister());
	}

	{
	__m128d r20 = _a.halfRowAsDouble(2u, true);
	__m128d r21 = _a.halfRowAsDouble(2u, false);
	second = _mm_cvtpd_ps(concat64_helper(r20, r21, _b, false));
	out.rows[2] = vectorSIMDf(_mm_cvtpd_ps(concat64_helper(r20, r21, _b, true))) | _mm_castps_si128((vectorSIMDf(_mm_movelh_ps(second, second)) & mask0011).getAsRegister());
	}

	{
	__m128d r30 = _a.halfRowAsDouble(3u, true);
	__m128d r31 = _a.halfRowAsDouble(3u, false);
	second = _mm_cvtpd_ps(concat64_helper(r30, r31, _b, false));
	out.rows[3] = vectorSIMDf(_mm_cvtpd_ps(concat64_helper(r30, r31, _b, true))) | _mm_castps_si128((vectorSIMDf(_mm_movelh_ps(second, second)) & mask0011).getAsRegister());
	}

	return out;
}

inline matrix4SIMD& matrix4SIMD::setScale(const core::vectorSIMDf& _scale)
{
	const __m128i mask0001 = BUILD_MASKF(0, 0, 0, 1);

	rows[0] = (_scale & BUILD_MASKF(1, 0, 0, 0)) | _mm_castps_si128((rows[0] & mask0001).getAsRegister());
	rows[1] = (_scale & BUILD_MASKF(0, 1, 0, 0)) | _mm_castps_si128((rows[1] & mask0001).getAsRegister());
	rows[2] = (_scale & BUILD_MASKF(0, 0, 1, 0)) | _mm_castps_si128((rows[2] & mask0001).getAsRegister());
	rows[3] = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

	return *this;
}

//! Returns last column of the matrix.
inline vectorSIMDf matrix4SIMD::getTranslation() const
{
	__m128 tmp1 = _mm_unpackhi_ps(rows[0].getAsRegister(), rows[1].getAsRegister()); // (0z,1z,0w,1w)
	__m128 tmp2 = _mm_unpackhi_ps(rows[2].getAsRegister(), rows[3].getAsRegister()); // (2z,3z,2w,3w)
	__m128 col3 = _mm_movehl_ps(tmp1, tmp2);// (0w,1w,2w,3w)

	return col3;
}
//! Returns translation part of the matrix (w component is always 0).
inline vectorSIMDf matrix4SIMD::getTranslation3D() const
{
	__m128 tmp1 = _mm_unpackhi_ps(rows[0].getAsRegister(), rows[1].getAsRegister()); // (0z,1z,0w,1w)
	__m128 tmp2 = _mm_unpackhi_ps(rows[2].getAsRegister(), _mm_setzero_ps()); // (2z,0,2w,0)
	__m128 transl = _mm_movehl_ps(tmp1, tmp2);// (0w,1w,2w,0)

	return transl;
}

inline bool matrix4SIMD::getInverseTransform(matrix4SIMD& _out) const
{
	auto mat2mul = [](vectorSIMDf _A, vectorSIMDf _B)
	{
		return _A*_B.xwxw()+_A.yxwz()*_B.zyzy();
	};
	auto mat2adjmul = [](vectorSIMDf _A, vectorSIMDf _B)
	{
		return _A.wwxx()*_B-_A.yyzz()*_B.zwxy();
	};
	auto mat2muladj = [](vectorSIMDf _A, vectorSIMDf _B)
	{
		return _A*_B.wxwx()-_A.yxwz()*_B.zyzy();
	};

	vectorSIMDf A = _mm_movelh_ps(rows[0].getAsRegister(), rows[1].getAsRegister());
	vectorSIMDf B = _mm_movehl_ps(rows[1].getAsRegister(), rows[0].getAsRegister());
	vectorSIMDf C = _mm_movelh_ps(rows[2].getAsRegister(), rows[3].getAsRegister());
	vectorSIMDf D = _mm_movehl_ps(rows[3].getAsRegister(), rows[2].getAsRegister());

	vectorSIMDf allDets =	vectorSIMDf(_mm_shuffle_ps(rows[0].getAsRegister(),rows[2].getAsRegister(),_MM_SHUFFLE(2,0,2,0)))*
							vectorSIMDf(_mm_shuffle_ps(rows[1].getAsRegister(),rows[3].getAsRegister(),_MM_SHUFFLE(3,1,3,1)))
						-
							vectorSIMDf(_mm_shuffle_ps(rows[0].getAsRegister(),rows[2].getAsRegister(),_MM_SHUFFLE(3,1,3,1)))*
							vectorSIMDf(_mm_shuffle_ps(rows[1].getAsRegister(),rows[3].getAsRegister(),_MM_SHUFFLE(2,0,2,0)));

	auto detA = allDets.xxxx();
	auto detB = allDets.yyyy();
	auto detC = allDets.zzzz();
	auto detD = allDets.wwww();

	// https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html
	auto D_C = mat2adjmul(D, C);
	// A#B
	auto A_B = mat2adjmul(A, B);
	// X# = |D|A - B(D#C)
	auto X_ = detD*A - mat2mul(B, D_C);
	// W# = |A|D - C(A#B)
	auto W_ = detA*D - mat2mul(C, A_B);

	// |M| = |A|*|D| + ... (continue later)
	auto detM = detA*detD;

	// Y# = |B|C - D(A#B)#
	auto Y_ = detB*C - mat2muladj(D, A_B);
	// Z# = |C|B - A(D#C)#
	auto Z_ = detC*B -  mat2muladj(A, D_C);

	// |M| = |A|*|D| + |B|*|C| ... (continue later)
	detM += detB*detC;

	// tr((A#B)(D#C))
	__m128 tr = (A_B*D_C.xzyw()).getAsRegister();
	tr = _mm_hadd_ps(tr, tr);
	tr = _mm_hadd_ps(tr, tr);
	// |M| = |A|*|D| + |B|*|C| - tr((A#B)(D#C)
	detM -= tr;

	if (core::iszero(detM.x, FLT_MIN))
		return false;

	// (1/|M|, -1/|M|, -1/|M|, 1/|M|)
	auto rDetM = vectorSIMDf(1.f, -1.f, -1.f, 1.f)*core::reciprocal(detM);

	X_ *= rDetM;
	Y_ *= rDetM;
	Z_ *= rDetM;
	W_ *= rDetM;

	// apply adjugate and store, here we combine adjugate shuffle and store shuffle
	_out.rows[0] = _mm_shuffle_ps(X_.getAsRegister(), Y_.getAsRegister(), _MM_SHUFFLE(1, 3, 1, 3));
	_out.rows[1] = _mm_shuffle_ps(X_.getAsRegister(), Y_.getAsRegister(), _MM_SHUFFLE(0, 2, 0, 2));
	_out.rows[2] = _mm_shuffle_ps(Z_.getAsRegister(), W_.getAsRegister(), _MM_SHUFFLE(1, 3, 1, 3));
	_out.rows[3] = _mm_shuffle_ps(Z_.getAsRegister(), W_.getAsRegister(), _MM_SHUFFLE(0, 2, 0, 2));

	return true;
}

inline vectorSIMDf matrix4SIMD::sub3x3TransformVect(const vectorSIMDf& _in) const
{
	matrix4SIMD cp{*this};
	vectorSIMDf out = _in & BUILD_MASKF(1, 1, 1, 0);
	transformVect(out);
	return out;
}

inline void matrix4SIMD::transformVect(vectorSIMDf& _out, const vectorSIMDf& _in) const
{
	vectorSIMDf r[4];
	for (size_t i = 0u; i < VectorCount; ++i)
		r[i] = rows[i] * _in;

	_out = _mm_hadd_ps(
		_mm_hadd_ps(r[0].getAsRegister(), r[1].getAsRegister()),
		_mm_hadd_ps(r[2].getAsRegister(), r[3].getAsRegister())
	);
}

inline matrix4SIMD matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
{
	const float h = core::reciprocal<float>(tanf(fieldOfViewRadians*0.5f));
	_NBL_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
	const float w = h / aspectRatio;

	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix4SIMD m;
	m.rows[0] = vectorSIMDf(w, 0.f, 0.f, 0.f);
	m.rows[1] = vectorSIMDf(0.f, -h, 0.f, 0.f);
	m.rows[2] = vectorSIMDf(0.f, 0.f, -zFar/(zFar-zNear), -zNear*zFar/(zFar-zNear));
	m.rows[3] = vectorSIMDf(0.f, 0.f, -1.f, 0.f);

	return m;
}
inline matrix4SIMD matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
{
	const float h = core::reciprocal<float>(tanf(fieldOfViewRadians*0.5f));
	_NBL_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
	const float w = h / aspectRatio;

	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix4SIMD m;
	m.rows[0] = vectorSIMDf(w, 0.f, 0.f, 0.f);
	m.rows[1] = vectorSIMDf(0.f, -h, 0.f, 0.f);
	m.rows[2] = vectorSIMDf(0.f, 0.f, zFar/(zFar-zNear), -zNear*zFar/(zFar-zNear));
	m.rows[3] = vectorSIMDf(0.f, 0.f, 1.f, 0.f);

	return m;
}

inline matrix4SIMD matrix4SIMD::buildProjectionMatrixOrthoRH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
{
	_NBL_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix4SIMD m;
	m.rows[0] = vectorSIMDf(2.f/widthOfViewVolume, 0.f, 0.f, 0.f);
	m.rows[1] = vectorSIMDf(0.f, -2.f/heightOfViewVolume, 0.f, 0.f);
	m.rows[2] = vectorSIMDf(0.f, 0.f, -1.f/(zFar-zNear), -zNear/(zFar-zNear));
	m.rows[3] = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

	return m;
}
inline matrix4SIMD matrix4SIMD::buildProjectionMatrixOrthoLH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
{
	_NBL_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix4SIMD m;
	m.rows[0] = vectorSIMDf(2.f/widthOfViewVolume, 0.f, 0.f, 0.f);
	m.rows[1] = vectorSIMDf(0.f, -2.f/heightOfViewVolume, 0.f, 0.f);
	m.rows[2] = vectorSIMDf(0.f, 0.f, 1.f/(zFar-zNear), -zNear/(zFar-zNear));
	m.rows[3] = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

	return m;
}



inline __m128d matrix4SIMD::halfRowAsDouble(size_t _n, bool _firstHalf) const
{
	return _mm_cvtps_pd(_firstHalf ? rows[_n].xyxx().getAsRegister() : rows[_n].zwxx().getAsRegister());
}
inline __m128d matrix4SIMD::concat64_helper(const __m128d& _a0, const __m128d& _a1, const matrix4SIMD& _mtx, bool _firstHalf)
{
	__m128d r0 = _mtx.halfRowAsDouble(0u, _firstHalf);
	__m128d r1 = _mtx.halfRowAsDouble(1u, _firstHalf);
	__m128d r2 = _mtx.halfRowAsDouble(2u, _firstHalf);
	__m128d r3 = _mtx.halfRowAsDouble(3u, _firstHalf);

	//const __m128d mask01 = _mm_castsi128_pd(_mm_setr_epi32(0, 0, 0xffffffff, 0xffffffff));

	__m128d res;
	res = _mm_mul_pd(_mm_shuffle_pd(_a0, _a0, 0), r0);
	res = _mm_add_pd(res, _mm_mul_pd(_mm_shuffle_pd(_a0, _a0, 3/*0b11*/), r1));
	res = _mm_add_pd(res, _mm_mul_pd(_mm_shuffle_pd(_a1, _a1, 0), r2));
	res = _mm_add_pd(res, _mm_mul_pd(_mm_shuffle_pd(_a1, _a1, 3/*0b11*/), r3));
	return res;
}

#undef BUILD_MASKF
#undef BROADCAST32
#else
#error "no implementation"
#endif

inline bool matrix4SIMD::isBoxInFrustum(const aabbox3d<float>& bbox)
{
	vectorSIMDf MinEdge, MaxEdge;
	MinEdge.set(bbox.MinEdge);
	MaxEdge.set(bbox.MaxEdge);
	MinEdge.w = 1.f;
	MaxEdge.w = 1.f;


	auto getClosestDP = [&MinEdge,&MaxEdge](const vectorSIMDf& toDot) -> float
	{
		return dot(mix(MaxEdge,MinEdge,toDot<vectorSIMDf(0.f)),toDot)[0];
	};

	// near plane
	if (getClosestDP(rows[3])<=0.f)
		return false;

	// x max
	if (getClosestDP(rows[3]+rows[0])<=0.f)
		return false;
	// y max
	if (getClosestDP(rows[3]+rows[1])<=0.f)
		return false;
	// x min
	if (getClosestDP(rows[3]-rows[0])<=0.f)
		return false;
	// y min
	if (getClosestDP(rows[3]-rows[1])<=0.f)
		return false;

	// far plane
	if (getClosestDP(rows[3]+rows[2])<=0.f)
		return false;

	return true;
}

}
} // nbl::core

#endif
