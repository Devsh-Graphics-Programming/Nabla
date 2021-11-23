// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_MATRIX3X4SIMD_H_INCLUDED__
#define __NBL_MATRIX3X4SIMD_H_INCLUDED__

#include "vectorSIMD.h"
#include "quaternion.h"

namespace nbl::core
{

class matrix4x3;

#define _NBL_MATRIX_ALIGNMENT _NBL_SIMD_ALIGNMENT
static_assert(_NBL_MATRIX_ALIGNMENT>=_NBL_VECTOR_ALIGNMENT,"Matrix must be equally or more aligned than vector!");

//! Equivalent of GLSL's mat4x3
class matrix3x4SIMD// : private AllocationOverrideBase<_NBL_MATRIX_ALIGNMENT> EBO inheritance problem w.r.t `rows[3]`
{
	public:
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t VectorCount = 3u;
		vectorSIMDf rows[VectorCount];

		explicit matrix3x4SIMD(	const vectorSIMDf& _r0 = vectorSIMDf(1.f, 0.f, 0.f, 0.f),
								const vectorSIMDf& _r1 = vectorSIMDf(0.f, 1.f, 0.f, 0.f),
								const vectorSIMDf& _r2 = vectorSIMDf(0.f, 0.f, 1.f, 0.f)) : rows{_r0, _r1, _r2}
		{
		}

		matrix3x4SIMD(	float _a00, float _a01, float _a02, float _a03,
						float _a10, float _a11, float _a12, float _a13,
						float _a20, float _a21, float _a22, float _a23)
								: matrix3x4SIMD(vectorSIMDf(_a00, _a01, _a02, _a03),
												vectorSIMDf(_a10, _a11, _a12, _a13),
												vectorSIMDf(_a20, _a21, _a22, _a23))
		{
		}

		explicit matrix3x4SIMD(const float* const _data)
		{
			if (!_data)
				return;
			for (size_t i = 0u; i < VectorCount; ++i)
				rows[i] = vectorSIMDf(_data + 4*i);
		}
		matrix3x4SIMD(const float* const _data, bool ALIGNED)
		{
			if (!_data)
				return;
			for (size_t i = 0u; i < VectorCount; ++i)
				rows[i] = vectorSIMDf(_data + 4*i, ALIGNED);
		}

		float* pointer() { return rows[0].pointer; }
		const float* pointer() const { return rows[0].pointer; }

		inline matrix3x4SIMD& set(const matrix4x3& _retarded);
		inline matrix4x3 getAsRetardedIrrlichtMatrix() const;

		static inline matrix3x4SIMD concatenateBFollowedByA(const matrix3x4SIMD& _a, const matrix3x4SIMD& _b);

		static inline matrix3x4SIMD concatenateBFollowedByAPrecisely(const matrix3x4SIMD& _a, const matrix3x4SIMD& _b);

		inline matrix3x4SIMD& concatenateAfter(const matrix3x4SIMD& _other)
		{
			return *this = concatenateBFollowedByA(*this, _other);
		}

		inline matrix3x4SIMD& concatenateBefore(const matrix3x4SIMD& _other)
		{
			return *this = concatenateBFollowedByA(_other, *this);
		}

		inline matrix3x4SIMD& concatenateAfterPrecisely(const matrix3x4SIMD& _other)
		{
			return *this = concatenateBFollowedByAPrecisely(*this, _other);
		}

		inline matrix3x4SIMD& concatenateBeforePrecisely(const matrix3x4SIMD& _other)
		{
			return *this = concatenateBFollowedByAPrecisely(_other, *this);
		}

		inline bool operator==(const matrix3x4SIMD& _other)
		{
			return !(*this != _other);
		}

		inline bool operator!=(const matrix3x4SIMD& _other);


		inline matrix3x4SIMD operator-() const
		{
			matrix3x4SIMD retval;
			retval.rows[0] = -rows[0];
			retval.rows[1] = -rows[1];
			retval.rows[2] = -rows[2];
			return retval;
		}


		inline matrix3x4SIMD& operator+=(const matrix3x4SIMD& _other);
		inline matrix3x4SIMD operator+(const matrix3x4SIMD& _other) const
		{
			matrix3x4SIMD retval(*this);
			return retval += _other;
		}

		inline matrix3x4SIMD& operator-=(const matrix3x4SIMD& _other);
		inline matrix3x4SIMD operator-(const matrix3x4SIMD& _other) const
		{
			matrix3x4SIMD retval(*this);
			return retval -= _other;
		}

		inline matrix3x4SIMD& operator*=(float _scalar);
		inline matrix3x4SIMD operator*(float _scalar) const
		{
			matrix3x4SIMD retval(*this);
			return retval *= _scalar;
		}

		inline matrix3x4SIMD& setTranslation(const vectorSIMDf& _translation)
		{
			// no faster way of doing it?
			rows[0].w = _translation.x;
			rows[1].w = _translation.y;
			rows[2].w = _translation.z;
			return *this;
		}
		inline vectorSIMDf getTranslation() const;
		inline vectorSIMDf getTranslation3D() const;

		inline matrix3x4SIMD& setScale(const vectorSIMDf& _scale);

		inline vectorSIMDf getScale() const;

		inline void transformVect(vectorSIMDf& _out, const vectorSIMDf& _in) const;
		inline void transformVect(vectorSIMDf& _in_out) const
		{
			transformVect(_in_out, _in_out);
		}

		inline void pseudoMulWith4x1(vectorSIMDf& _out, const vectorSIMDf& _in) const;
		inline void pseudoMulWith4x1(vectorSIMDf& _in_out) const
		{
			pseudoMulWith4x1(_in_out,_in_out);
		}

		inline void mulSub3x3WithNx1(vectorSIMDf& _out, const vectorSIMDf& _in) const;
		inline void mulSub3x3WithNx1(vectorSIMDf& _in_out) const
		{
			mulSub3x3WithNx1(_in_out, _in_out);
		}

		inline static matrix3x4SIMD buildCameraLookAtMatrixLH(
			const vectorSIMDf& position,
			const vectorSIMDf& target,
			const vectorSIMDf& upVector);
		inline static matrix3x4SIMD buildCameraLookAtMatrixRH(
			const vectorSIMDf& position,
			const vectorSIMDf& target,
			const vectorSIMDf& upVector);

		inline matrix3x4SIMD& setRotation(const quaternion& _quat);

		inline matrix3x4SIMD& setScaleRotationAndTranslation(	const vectorSIMDf& _scale,
																const quaternion& _quat,
																const vectorSIMDf& _translation);

		inline vectorSIMDf getPseudoDeterminant() const
		{
			vectorSIMDf tmp;
			return determinant_helper(tmp);
		}

		inline bool getInverse(matrix3x4SIMD& _out) const;
		bool makeInverse()
		{
			matrix3x4SIMD tmp;

			if (getInverse(tmp))
			{
				*this = tmp;
				return true;
			}
			return false;
		}

		//
		inline bool getSub3x3InverseTranspose(matrix3x4SIMD& _out) const;

		//
		inline bool getSub3x3InverseTransposePacked(float outRows[9]) const
		{
			matrix3x4SIMD tmp;
			if (!getSub3x3InverseTranspose(tmp))
				return false;

			float* _out = outRows;
			for (auto i=0; i<3; i++)
			{
				const auto& row = tmp.rows[i];
				for (auto j=0; j<3; j++)
					*(_out++) = row[j];
			}

			return true;
		}

		//
		inline core::matrix3x4SIMD getSub3x3TransposeCofactors() const;

		//
		inline void setTransformationCenter(const vectorSIMDf& _center, const vectorSIMDf& _translation);

		//
		static inline matrix3x4SIMD buildAxisAlignedBillboard(
			const vectorSIMDf& camPos,
			const vectorSIMDf& center,
			const vectorSIMDf& translation,
			const vectorSIMDf& axis,
			const vectorSIMDf& from);


		//
		float& operator()(size_t _i, size_t _j) { return rows[_i].pointer[_j]; }
		const float& operator()(size_t _i, size_t _j) const { return rows[_i].pointer[_j]; }

		//
		inline const vectorSIMDf& operator[](size_t _rown) const { return rows[_rown]; }
		inline vectorSIMDf& operator[](size_t _rown) { return rows[_rown]; }

	private:
		static inline vectorSIMDf doJob(const __m128& a, const matrix3x4SIMD& _mtx);

		// really need that dvec<2> or wider
		inline __m128d halfRowAsDouble(size_t _n, bool _0) const;
		static inline __m128d doJob_d(const __m128d& _a0, const __m128d& _a1, const matrix3x4SIMD& _mtx, bool _xyHalf);

		vectorSIMDf determinant_helper(vectorSIMDf& r1crossr2) const
		{
			r1crossr2 = core::cross(rows[1], rows[2]);
			return core::dot(rows[0], r1crossr2);
		}
};

inline matrix3x4SIMD concatenateBFollowedByA(const matrix3x4SIMD& _a, const matrix3x4SIMD& _b)
{
    return matrix3x4SIMD::concatenateBFollowedByA(_a, _b);
}
/*
inline matrix3x4SIMD concatenateBFollowedByAPrecisely(const matrix3x4SIMD& _a, const matrix3x4SIMD& _b)
{
    return matrix3x4SIMD::concatenateBFollowedByAPrecisely(_a, _b);
}
*/

}

#endif
