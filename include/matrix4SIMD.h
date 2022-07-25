// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_MATRIX4SIMD_H_INCLUDED__
#define __NBL_MATRIX4SIMD_H_INCLUDED__

#include "matrix3x4SIMD.h"

namespace nbl
{
namespace core
{

template<typename T>
class aabbox3d;


class NBL_API matrix4SIMD// : public AlignedBase<_NBL_SIMD_ALIGNMENT> don't inherit from AlignedBase (which is empty) because member `rows[4]` inherits from it as well
{
	public:
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t VectorCount = 4u;
		vectorSIMDf rows[VectorCount];

		inline explicit matrix4SIMD(const vectorSIMDf& _r0 = vectorSIMDf(1.f, 0.f, 0.f, 0.f),
									const vectorSIMDf& _r1 = vectorSIMDf(0.f, 1.f, 0.f, 0.f),
									const vectorSIMDf& _r2 = vectorSIMDf(0.f, 0.f, 1.f, 0.f),
									const vectorSIMDf& _r3 = vectorSIMDf(0.f, 0.f, 0.f, 1.f))
											: rows{ _r0, _r1, _r2, _r3 }
		{
		}

		inline matrix4SIMD(	float _a00, float _a01, float _a02, float _a03,
							float _a10, float _a11, float _a12, float _a13,
							float _a20, float _a21, float _a22, float _a23,
							float _a30, float _a31, float _a32, float _a33)
									: matrix4SIMD(	vectorSIMDf(_a00, _a01, _a02, _a03),
													vectorSIMDf(_a10, _a11, _a12, _a13),
													vectorSIMDf(_a20, _a21, _a22, _a23),
													vectorSIMDf(_a30, _a31, _a32, _a33))
		{
		}

		inline explicit matrix4SIMD(const float* const _data)
		{
			if (!_data)
				return;
			for (size_t i = 0u; i < VectorCount; ++i)
				rows[i] = vectorSIMDf(_data + 4 * i);
		}
		inline matrix4SIMD(const float* const _data, bool ALIGNED)
		{
			if (!_data)
				return;
			for (size_t i = 0u; i < VectorCount; ++i)
				rows[i] = vectorSIMDf(_data + 4 * i, ALIGNED);
		}

		inline explicit matrix4SIMD(const matrix3x4SIMD& smallMat)
		{
			*reinterpret_cast<matrix3x4SIMD*>(this) = smallMat;
			rows[3].set(0.f,0.f,0.f,1.f);
		}

		inline matrix3x4SIMD extractSub3x4() const
		{
			return matrix3x4SIMD(rows[0],rows[1],rows[2]);
		}

		//! Access by row
		inline const vectorSIMDf& getRow(size_t _rown) const{ return rows[_rown]; }
		inline vectorSIMDf& getRow(size_t _rown) { return rows[_rown]; }

		//! Access by element
		inline float operator()(size_t _i, size_t _j) const { return rows[_i].pointer[_j]; }
		inline float& operator()(size_t _i, size_t _j) { return rows[_i].pointer[_j]; }

		//! Access for memory
		inline const float* pointer() const {return rows[0].pointer;}
		inline float* pointer() {return rows[0].pointer;}


		inline bool operator==(const matrix4SIMD& _other) const
		{
			return !(*this != _other);
		}
		inline bool operator!=(const matrix4SIMD& _other) const;

		inline matrix4SIMD& operator+=(const matrix4SIMD& _other);
		inline matrix4SIMD operator+(const matrix4SIMD& _other) const
		{
			matrix4SIMD r{*this};
			return r += _other;
		}

		inline matrix4SIMD& operator-=(const matrix4SIMD& _other);
		inline matrix4SIMD operator-(const matrix4SIMD& _other) const
		{
			matrix4SIMD r{*this};
			return r -= _other;
		}

		inline matrix4SIMD& operator*=(float _scalar);
		inline matrix4SIMD operator*(float _scalar) const
		{
			matrix4SIMD r{*this};
			return r *= _scalar;
		}

		static inline matrix4SIMD concatenateBFollowedByA(const matrix4SIMD& _a, const matrix4SIMD& _b);
		static inline matrix4SIMD concatenateBFollowedByAPrecisely(const matrix4SIMD& _a, const matrix4SIMD& _b);

		inline bool isIdentity() const
		{
			return *this == matrix4SIMD();
		}
		inline bool isIdentity(float _tolerance) const;

		inline bool isOrthogonal() const
		{
			return concatenateBFollowedByA(transpose(*this), *this).isIdentity();
		}
		inline bool isOrthogonal(float _tolerance) const
		{
			return concatenateBFollowedByA(transpose(*this), *this).isIdentity(_tolerance);
		}

		inline matrix4SIMD& setScale(const core::vectorSIMDf& _scale);
		inline matrix4SIMD& setScale(float _scale)
		{
			return setScale(vectorSIMDf(_scale));
		}

		inline void setTranslation(const float* _t)
		{
			for (size_t i = 0u; i < 3u; ++i)
				rows[i].w = _t[i];
		}
		//! Takes into account only x,y,z components of _t
		inline void setTranslation(const vectorSIMDf& _t)
		{
			setTranslation(_t.pointer);
		}
		inline void setTranslation(const vector3d<float>& _t)
		{
			setTranslation(&_t.X);
		}

		//! Returns last column of the matrix.
		inline vectorSIMDf getTranslation() const;

		//! Returns translation part of the matrix (w component is always 0).
		inline vectorSIMDf getTranslation3D() const;
		
		enum class E_MATRIX_INVERSE_PRECISION
		{
		  EMIP_FAST_RECIPROCAL,
		  EMIP_32BIT,
		  EMIP_64BBIT
		};

		template<E_MATRIX_INVERSE_PRECISION precision = E_MATRIX_INVERSE_PRECISION::EMIP_FAST_RECIPROCAL>
		inline bool getInverseTransform(matrix4SIMD& _out) const
		{
			if constexpr (precision == E_MATRIX_INVERSE_PRECISION::EMIP_64BBIT)
			{
				double a = rows[0][0], b = rows[0][1], c = rows[0][2], d = rows[0][3];
				double e = rows[1][0], f = rows[1][1], g = rows[1][2], h = rows[1][3];
				double i = rows[2][0], j = rows[2][1], k = rows[2][2], l = rows[2][3];
				double m = rows[3][0], n = rows[3][1], o = rows[3][2], p = rows[3][3];

				double kp_lo = k * p - l * o;
				double jp_ln = j * p - l * n;
				double jo_kn = j * o - k * n;
				double ip_lm = i * p - l * m;
				double io_km = i * o - k * m;
				double in_jm = i * n - j * m;

				double a11 = +(f * kp_lo - g * jp_ln + h * jo_kn);
				double a12 = -(e * kp_lo - g * ip_lm + h * io_km);
				double a13 = +(e * jp_ln - f * ip_lm + h * in_jm);
				double a14 = -(e * jo_kn - f * io_km + g * in_jm);

				double det = a * a11 + b * a12 + c * a13 + d * a14;

				if (core::iszero(det, DBL_MIN))
					return false;

				double invDet = 1.0 / det;

				_out.rows[0][0] = a11 * invDet;
				_out.rows[1][0] = a12 * invDet;
				_out.rows[2][0] = a13 * invDet;
				_out.rows[3][0] = a14 * invDet;

				_out.rows[0][1] = -(b * kp_lo - c * jp_ln + d * jo_kn) * invDet;
				_out.rows[1][1] = +(a * kp_lo - c * ip_lm + d * io_km) * invDet;
				_out.rows[2][1] = -(a * jp_ln - b * ip_lm + d * in_jm) * invDet;
				_out.rows[3][1] = +(a * jo_kn - b * io_km + c * in_jm) * invDet;

				double gp_ho = g * p - h * o;
				double fp_hn = f * p - h * n;
				double fo_gn = f * o - g * n;
				double ep_hm = e * p - h * m;
				double eo_gm = e * o - g * m;
				double en_fm = e * n - f * m;

				_out.rows[0][2] = +(b * gp_ho - c * fp_hn + d * fo_gn) * invDet;
				_out.rows[1][2] = -(a * gp_ho - c * ep_hm + d * eo_gm) * invDet;
				_out.rows[2][2] = +(a * fp_hn - b * ep_hm + d * en_fm) * invDet;
				_out.rows[3][2] = -(a * fo_gn - b * eo_gm + c * en_fm) * invDet;

				double gl_hk = g * l - h * k;
				double fl_hj = f * l - h * j;
				double fk_gj = f * k - g * j;
				double el_hi = e * l - h * i;
				double ek_gi = e * k - g * i;
				double ej_fi = e * j - f * i;

				_out.rows[0][3] = -(b * gl_hk - c * fl_hj + d * fk_gj) * invDet;
				_out.rows[1][3] = +(a * gl_hk - c * el_hi + d * ek_gi) * invDet;
				_out.rows[2][3] = -(a * fl_hj - b * el_hi + d * ej_fi) * invDet;
				_out.rows[3][3] = +(a * fk_gj - b * ek_gi + c * ej_fi) * invDet;

				return true;
			}
			else
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

				vectorSIMDf rDetM;

				// (1/|M|, -1/|M|, -1/|M|, 1/|M|)
				if constexpr (precision == E_MATRIX_INVERSE_PRECISION::EMIP_FAST_RECIPROCAL)
					rDetM = vectorSIMDf(1.f, -1.f, -1.f, 1.f)*core::reciprocal(detM);
				else if constexpr (precision == E_MATRIX_INVERSE_PRECISION::EMIP_32BIT)
					rDetM = vectorSIMDf(1.f, -1.f, -1.f, 1.f).preciseDivision(detM);

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
		}

		inline vectorSIMDf sub3x3TransformVect(const vectorSIMDf& _in) const;

		inline void transformVect(vectorSIMDf& _out, const vectorSIMDf& _in) const;
		inline void transformVect(vectorSIMDf& _vector) const
		{
			transformVect(_vector, _vector);
		}

		inline void translateVect(vectorSIMDf& _vect) const
		{
			_vect += getTranslation();
		}

		bool isBoxInFrustum(const aabbox3d<float>& bbox);

		bool perspectiveTransformVect(core::vectorSIMDf& inOutVec)
		{
			transformVect(inOutVec);
			const bool inFront = inOutVec[3] > 0.f;
			inOutVec /= inOutVec.wwww();
			return inFront;
		}

		core::vector2di fragCoordTransformVect(const core::vectorSIMDf& _in, const core::dimension2du& viewportDimensions)
		{
			core::vectorSIMDf pos(_in);
			pos.w = 1.f;
			if (perspectiveTransformVect(pos))
				core::vector2di(-0x80000000, -0x80000000);

			pos[0] *= 0.5f;
			pos[1] *= 0.5f;
			pos[0] += 0.5f;
			pos[1] += 0.5f;

			return core::vector2di(pos[0] * float(viewportDimensions.Width), pos[1] * float(viewportDimensions.Height));
		}

		static inline matrix4SIMD buildProjectionMatrixPerspectiveFovRH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar);
		static inline matrix4SIMD buildProjectionMatrixPerspectiveFovLH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar);

		static inline matrix4SIMD buildProjectionMatrixOrthoRH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar);
		static inline matrix4SIMD buildProjectionMatrixOrthoLH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar);

		//! Access by row
		inline const vectorSIMDf& operator[](size_t _rown) const { return rows[_rown]; }
		//! Access by row
		inline vectorSIMDf& operator[](size_t _rown) { return rows[_rown]; }

	private:
		//! TODO: implement a dvec<2>
		inline __m128d halfRowAsDouble(size_t _n, bool _firstHalf) const;
		static inline __m128d concat64_helper(const __m128d& _a0, const __m128d& _a1, const matrix4SIMD& _mtx, bool _firstHalf);
};

inline matrix4SIMD operator*(float _scalar, const matrix4SIMD& _mtx)
{
    return _mtx * _scalar;
}

inline matrix4SIMD concatenateBFollowedByA(const matrix4SIMD& _a, const matrix4SIMD& _b)
{
    return matrix4SIMD::concatenateBFollowedByA(_a, _b);
}
/*
inline matrix4SIMD concatenateBFollowedByAPrecisely(const matrix4SIMD& _a, const matrix4SIMD& _b)
{
    return matrix4SIMD::concatenateBFollowedByAPrecisely(_a, _b);
}
*/


}} // nbl::core

#endif
