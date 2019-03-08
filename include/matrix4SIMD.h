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
    vectorSIMDf rows[4];

#define BUILD_MASKF(_x_, _y_, _z_, _w_) _mm_setr_epi32(_x_*0xffffffff, _y_*0xffffffff, _z_*0xffffffff, _w_*0xffffffff)
public:
    inline explicit matrix4SIMD(const vectorSIMDf& _r0 = vectorSIMDf(1.f, 0.f, 0.f, 0.f), const vectorSIMDf& _r1 = vectorSIMDf(0.f, 1.f, 0.f, 0.f), const vectorSIMDf& _r2 = vectorSIMDf(0.f, 0.f, 1.f, 0.f), const vectorSIMDf& _r3 = vectorSIMDf(0.f, 0.f, 0.f, 1.f))
        : rows{ _r0, _r1, _r2, _r3 } {}

    inline matrix4SIMD(
        float _a00, float _a01, float _a02, float _a03,
        float _a10, float _a11, float _a12, float _a13,
        float _a20, float _a21, float _a22, float _a23,
        float _a30, float _a31, float _a32, float _a33)
        : matrix4SIMD(vectorSIMDf(_a00, _a01, _a02, _a03), vectorSIMDf(_a10, _a11, _a12, _a13), vectorSIMDf(_a20, _a21, _a22, _a23), vectorSIMDf(_a30, _a31, _a32, _a33))
    {
    }

    inline explicit matrix4SIMD(const float* const _data)
    {
        if (!_data)
            return;
        for (size_t i = 0u; i < 4u; ++i)
            rows[i] = vectorSIMDf(_data + 4 * i);
    }
    inline matrix4SIMD(const float* const _data, bool ALIGNED)
    {
        if (!_data)
            return;
        for (size_t i = 0u; i < 4u; ++i)
            rows[i] = vectorSIMDf(_data + 4 * i, ALIGNED);
    }

    inline explicit matrix4SIMD(const matrix3x4SIMD& smallMat)
    {
        *reinterpret_cast<matrix3x4SIMD*>(this) = smallMat;
        rows[3].set(0.f,0.f,0.f,1.f);
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
    inline bool operator!=(const matrix4SIMD& _other) const
    {
        for (size_t i = 0u; i < 4u; ++i)
            if ((rows[i] != _other.rows[i]).any())
                return true;
        return false;
    }

    inline matrix4SIMD& operator+=(const matrix4SIMD& _other)
    {
        for (size_t i = 0u; i < 4u; ++i)
            rows[i] += _other.rows[i];
        return *this;
    }
    inline matrix4SIMD operator+(const matrix4SIMD& _other) const
    {
        matrix4SIMD r{*this};
        return r += _other;
    }

    inline matrix4SIMD& operator-=(const matrix4SIMD& _other)
    {
        for (size_t i = 0u; i < 4u; ++i)
            rows[i] -= _other.rows[i];
        return *this;
    }
    inline matrix4SIMD operator-(const matrix4SIMD& _other) const
    {
        matrix4SIMD r{*this};
        return r -= _other;
    }

    inline matrix4SIMD& operator*=(float _scalar)
    {
        for (size_t i = 0u; i < 4u; ++i)
            rows[i] *= _scalar;
        return *this;
    }
    inline matrix4SIMD operator*(float _scalar) const
    {
        matrix4SIMD r{*this};
        return r *= _scalar;
    }

#define BROADCAST32(fpx) _MM_SHUFFLE(fpx, fpx, fpx, fpx)
    static inline matrix4SIMD concatenateBFollowedByA(const matrix4SIMD& _a, const matrix4SIMD& _b)
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
    static inline matrix4SIMD concatenateBFollowedByAPrecisely(const matrix4SIMD& _a, const matrix4SIMD& _b)
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

    inline bool isIdentity() const
    {
        return *this == matrix4SIMD();
    }
    inline bool isIdentity(float _tolerance) const
    {
        return this->equals(matrix4SIMD(), _tolerance);
    }

    inline bool isOrthogonal() const
    {
        return concatenateBFollowedByA(getTransposed(), *this).isIdentity();
    }
    inline bool isOrthogonal(float _tolerance) const
    {
        return concatenateBFollowedByA(getTransposed(), *this).isIdentity(_tolerance);
    }

    inline matrix4SIMD& setScale(const core::vectorSIMDf& _scale)
    {
        const __m128i mask0001 = BUILD_MASKF(0, 0, 0, 1);

        rows[0] = (_scale & BUILD_MASKF(1, 0, 0, 0)) | _mm_castps_si128((rows[0] & mask0001).getAsRegister());
        rows[1] = (_scale & BUILD_MASKF(0, 1, 0, 0)) | _mm_castps_si128((rows[1] & mask0001).getAsRegister());
        rows[2] = (_scale & BUILD_MASKF(0, 0, 1, 0)) | _mm_castps_si128((rows[2] & mask0001).getAsRegister());
        rows[3] = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

        return *this;
    }
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
    inline vectorSIMDf getTranslation() const
    {
        __m128 tmp1 = _mm_unpackhi_ps(rows[0].getAsRegister(), rows[1].getAsRegister()); // (0z,1z,0w,1w)
        __m128 tmp2 = _mm_unpackhi_ps(rows[2].getAsRegister(), rows[3].getAsRegister()); // (2z,3z,2w,3w)
        __m128 col3 = _mm_movehl_ps(tmp1, tmp2);// (0w,1w,2w,3w)

        return col3;
    }
    //! Returns translation part of the matrix (w component is always 0).
    inline vectorSIMDf getTranslation3D() const
    {
        __m128 tmp1 = _mm_unpackhi_ps(rows[0].getAsRegister(), rows[1].getAsRegister()); // (0z,1z,0w,1w)
        __m128 tmp2 = _mm_unpackhi_ps(rows[2].getAsRegister(), _mm_setzero_ps()); // (2z,0,2w,0)
        __m128 transl = _mm_movehl_ps(tmp1, tmp2);// (0w,1w,2w,0)

        return transl;
    }

    inline bool getInverseTransform(matrix4SIMD& _out) const
    {
        vectorSIMDf c0 = rows[0], c1 = rows[1], c2 = rows[2], c3 = vectorSIMDf(0.f, 0.f, 0.f, 1.f);
        core::transpose4(c0, c1, c2, c3);

        const vectorSIMDf c1crossc2 = c1.crossProduct(c2);

        const vectorSIMDf d = c0.dotProduct(c1crossc2);

        if (core::iszero(d.x, FLT_MIN))
            return false;

        _out.rows[0] = c1crossc2 / d;
        _out.rows[1] = (c2.crossProduct(c0)) / d;
        _out.rows[2] = (c0.crossProduct(c1)) / d;

        vectorSIMDf outC3 = vectorSIMDf(0.f, 0.f, 0.f, 1.f);
        core::transpose4(_out.rows[0], _out.rows[1], _out.rows[2], outC3);

        __m128i mask1110 = BUILD_MASKF(1, 1, 1, 0);
        vectorSIMDf r0 = (rows[0] * c3) & mask1110,
            r1 = (rows[1] * c3) & mask1110,
            r2 = (rows[2] * c3) & mask1110,
            r3 = vectorSIMDf(0.f);

        outC3 = _mm_hadd_ps(
            _mm_hadd_ps(r0.getAsRegister(), r1.getAsRegister()),
            _mm_hadd_ps(r2.getAsRegister(), r3.getAsRegister())
        );
        outC3 = -outC3;
        outC3.w = 1.f;
        core::transpose4(_out.rows[0], _out.rows[1], _out.rows[2], outC3);

        return true;
    }

    inline vectorSIMDf sub3x3TransformVect(const vectorSIMDf& _in) const
    {
        matrix4SIMD cp{*this};
        vectorSIMDf out = _in & BUILD_MASKF(1, 1, 1, 0);
        transformVect(out);
        return out;
    }

    inline void transformVect(vectorSIMDf& _out, const vectorSIMDf& _in) const
    {
        vectorSIMDf r[4];
        for (size_t i = 0u; i < 4u; ++i)
            r[i] = rows[i] * _in;

        _out = _mm_hadd_ps(
            _mm_hadd_ps(r[0].getAsRegister(), r[1].getAsRegister()),
            _mm_hadd_ps(r[2].getAsRegister(), r[3].getAsRegister())
        );
    }
    inline void transformVect(vectorSIMDf& _vector) const
    {
        transformVect(_vector, _vector);
    }

    inline void translateVect(vectorSIMDf& _vect) const
    {
        _vect += getTranslation();
    }

    static inline matrix4SIMD buildProjectionMatrixPerspectiveFovRH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
    {
        const double h = core::reciprocal(tan(fieldOfViewRadians*0.5));
        _IRR_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
        const float w = h / aspectRatio;

        _IRR_DEBUG_BREAK_IF(zNear == zFar); //division by zero

        matrix4SIMD m;
        m.rows[0] = vectorSIMDf(w, 0.f, 0.f, 0.f);
        m.rows[1] = vectorSIMDf(0.f, -h, 0.f, 0.f);
        m.rows[2] = vectorSIMDf(0.f, 0.f, zFar/(zFar-zNear), -zNear*zFar/(zFar-zNear));
        m.rows[3] = vectorSIMDf(0.f, 0.f, 1.f, 0.f);

        return m;
    }
    static inline matrix4SIMD buildProjectionMatrixPerspectiveFovLH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
    {
        const double h = core::reciprocal(tan(fieldOfViewRadians*0.5));
        _IRR_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
        const float w = h / aspectRatio;

        _IRR_DEBUG_BREAK_IF(zNear == zFar); //division by zero

        matrix4SIMD m;
        m.rows[0] = vectorSIMDf(w, 0.f, 0.f, 0.f);
        m.rows[1] = vectorSIMDf(0.f, (float)h, 0.f, 0.f);
        m.rows[2] = vectorSIMDf(0.f, 0.f, zFar/(zFar-zNear), -zNear*zFar/(zFar-zNear));
        m.rows[3] = vectorSIMDf(0.f, 0.f, 1.f, 0.f);

        return m;
    }

    static inline matrix4SIMD buildProjectionMatrixOrthoRH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
    {
        _IRR_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(zNear == zFar); //division by zero

        matrix4SIMD m;
        m.rows[0] = vectorSIMDf(2.f/widthOfViewVolume, 0.f, 0.f, 0.f);
        m.rows[1] = vectorSIMDf(0.f, -2.f/heightOfViewVolume, 0.f, 0.f);
        m.rows[2] = vectorSIMDf(0.f, 0.f, 1.f/(zFar-zNear), -zNear/(zFar-zNear));
        m.rows[3] = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

        return m;
    }
    static inline matrix4SIMD buildProjectionMatrixOrthoLH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
    {
        _IRR_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(zNear == zFar); //division by zero

        matrix4SIMD m;
        m.rows[0] = vectorSIMDf(2.f/widthOfViewVolume, 0.f, 0.f, 0.f);
        m.rows[1] = vectorSIMDf(0.f, 2.f/heightOfViewVolume, 0.f, 0.f);
        m.rows[2] = vectorSIMDf(0.f, 0.f, 1.f/(zFar-zNear), -zNear/(zFar-zNear));
        m.rows[3] = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

        return m;
    }

    static inline matrix4SIMD buildCameraLookAtMatrixLH(
        const core::vectorSIMDf& position,
        const core::vectorSIMDf& target,
        const core::vectorSIMDf& upVector)
    {
        const core::vectorSIMDf zaxis = core::normalize(target - position);
        const core::vectorSIMDf xaxis = core::normalize(upVector.crossProduct(zaxis));
        const core::vectorSIMDf yaxis = zaxis.crossProduct(xaxis);

        matrix4SIMD r;
        r.rows[0] = xaxis;
        r.rows[1] = yaxis;
        r.rows[2] = zaxis;
        r.rows[0].w = -xaxis.dotProductAsFloat(position);
        r.rows[1].w = -yaxis.dotProductAsFloat(position);
        r.rows[2].w = -zaxis.dotProductAsFloat(position);
        r.rows[3] = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

        return r;
    }
    static inline matrix4SIMD buildCameraLookAtMatrixRH(
        const core::vectorSIMDf& position,
        const core::vectorSIMDf& target,
        const core::vectorSIMDf& upVector)
    {
        const core::vectorSIMDf zaxis = core::normalize(position - target);
        const core::vectorSIMDf xaxis = core::normalize(upVector.crossProduct(zaxis));
        const core::vectorSIMDf yaxis = zaxis.crossProduct(xaxis);

        matrix4SIMD r;
        r.rows[0] = xaxis;
        r.rows[1] = yaxis;
        r.rows[2] = zaxis;
        r.rows[0].w = -xaxis.dotProductAsFloat(position);
        r.rows[1].w = -yaxis.dotProductAsFloat(position);
        r.rows[2].w = -zaxis.dotProductAsFloat(position);
        r.rows[3] = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

        return r;
    }

    inline matrix4SIMD getTransposed() const
    {
        matrix4SIMD r{*this};
        core::transpose4(r.rows);
        return r;
    }
    inline void getTransposed(matrix4SIMD& _out) const
    {
        _out = getTransposed();
    }

    inline bool equals(const matrix4SIMD& _other, float _tolerance) const
    {
        for (size_t i = 0u; i < 4u; ++i)
            if (!core::equals(rows[i], _other.rows[i], _tolerance).all())
                return false;
        return true;
    }

    static inline matrix4SIMD mix(const matrix4SIMD& _a, const matrix4SIMD& _b, float _x)
    {
        matrix4SIMD m;
        for (size_t i = 0u; i < 4u; ++i)
            m[i] = core::mix(_a[i], _b[i], vectorSIMDf(_x));
        return m;
    }

private:
    //! Access by row
    inline const vectorSIMDf& operator[](size_t _rown) const { return rows[_rown]; }
    //! Access by row
    inline vectorSIMDf& operator[](size_t _rown) { return rows[_rown]; }

    inline __m128d halfRowAsDouble(size_t _n, bool _firstHalf) const
    {
        return _mm_cvtps_pd(_firstHalf ? rows[_n].xyxx().getAsRegister() : rows[_n].zwxx().getAsRegister());
    }
    static inline __m128d concat64_helper(const __m128d& _a0, const __m128d& _a1, const matrix4SIMD& _mtx, bool _firstHalf)
    {
        __m128d r0 = _mtx.halfRowAsDouble(0u, _firstHalf);
        __m128d r1 = _mtx.halfRowAsDouble(1u, _firstHalf);
        __m128d r2 = _mtx.halfRowAsDouble(2u, _firstHalf);
        __m128d r3 = _mtx.halfRowAsDouble(3u, _firstHalf);

        const __m128d mask01 = _mm_castsi128_pd(_mm_setr_epi32(0, 0, 0xffffffff, 0xffffffff));

        __m128d res;
        res = _mm_mul_pd(_mm_shuffle_pd(_a0, _a0, 0), r0);
        res = _mm_add_pd(res, _mm_mul_pd(_mm_shuffle_pd(_a0, _a0, 3/*0b11*/), r1));
        res = _mm_add_pd(res, _mm_mul_pd(_mm_shuffle_pd(_a1, _a1, 0), r2));
        res = _mm_add_pd(res, _mm_mul_pd(_mm_shuffle_pd(_a1, _a1, 3/*0b11*/), r3));
        return res;
    }

#undef BUILD_MASKF
#undef BROADCAST32
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
inline matrix4SIMD mix(const matrix4SIMD& _a, const matrix4SIMD& _b, float _x)
{
    return matrix4SIMD::mix(_a, _b, _x);
}

inline matrix4SIMD lerp(const matrix4SIMD& _a, const matrix4SIMD& _b, float _x)
{
    return mix(_a, _b, _x);
}


}} // irr::core

#endif // __IRR_MATRIX4SIMD_H_INCLUDED__
