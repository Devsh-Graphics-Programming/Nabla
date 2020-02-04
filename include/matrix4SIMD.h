#ifndef __IRR_MATRIX4SIMD_H_INCLUDED__
#define __IRR_MATRIX4SIMD_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "matrix3x4SIMD.h"

namespace irr
{
namespace core
{


class matrix4SIMD// : public AlignedBase<_IRR_SIMD_ALIGNMENT> don't inherit from AlignedBase (which is empty) because member `rows[4]` inherits from it as well
{
	public:
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t VectorCount = 4u;
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

		inline bool getInverseTransform(matrix4SIMD& _out) const;

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

		static inline matrix4SIMD buildProjectionMatrixPerspectiveFovRH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar);
		static inline matrix4SIMD buildProjectionMatrixPerspectiveFovLH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar);

		static inline matrix4SIMD buildProjectionMatrixOrthoRH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar);
		static inline matrix4SIMD buildProjectionMatrixOrthoLH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar);

	private:
		//! Access by row
		inline const vectorSIMDf& operator[](size_t _rown) const { return rows[_rown]; }
		//! Access by row
		inline vectorSIMDf& operator[](size_t _rown) { return rows[_rown]; }

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


}} // irr::core

#endif // __IRR_MATRIX4SIMD_H_INCLUDED__
