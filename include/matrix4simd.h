#ifndef __IRR_MATRIX4SIMD_H_INCLUDED__
#define __IRR_MATRIX4SIMD_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "quaternion.h"

namespace irr { namespace core
{

#ifdef _IRR_WINDOWS_
    __declspec(align(SIMD_ALIGNMENT))
#endif
class matrix4SIMD
{
    vectorSIMDf rows[4];

#define BUILD_MASKF(_x_, _y_, _z_, _w_) _mm_castsi128_ps(_mm_setr_epi32(_x_*0xffffffff, _y_*0xffffffff, _z_*0xffffffff, _w_*0xffffffff))
public:
    inline matrix4SIMD() :
        rows{vectorSIMDf(1.f, 0.f, 0.f, 0.f), vectorSIMDf(0.f, 1.f, 0.f, 0.f), vectorSIMDf(0.f, 0.f, 1.f, 0.f), vectorSIMDf(0.f, 0.f, 0.f, 1.f)}
    {}

    //! Access by row
    inline const vectorSIMDf& operator[](size_t _rown) const { return rows[_rown]; }
    //! Access by row
    inline vectorSIMDf& operator[](size_t _rown) { return rows[_rown]; }

    //! Access by element
    inline float operator()(size_t _i, size_t _j) const { return rows[_i].pointer[_j]; }
    //! Access by element
    inline float& operator()(size_t _i, size_t _j) { return rows[_i].pointer[_j]; }

    inline bool operator==(const matrix4SIMD& _other) const
    {
        for (size_t i = 0u; i < 4u; ++i)
            if ((rows[i] != _other.rows[i]).all())
                return false;
        return true;
    }
    inline bool operator!=(const matrix4SIMD& _other) const
    {
        return !(*this == _other);
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

    inline matrix4SIMD& operator*=(const matrix4SIMD& _other)
    {
        auto calcRow = [&_other](const vectorSIMDf& _v)
        {
            __m128 v = _v.getAsRegister();

            v = _mm_mul_ps(_mm_shuffle_ps(v, v, 0x00), _other.rows[0].getAsRegister());
            v = _mm_add_ps(v, _mm_mul_ps(_mm_shuffle_ps(v, v, 0x55), _other.rows[1].getAsRegister()));
            v = _mm_add_ps(v, _mm_mul_ps(_mm_shuffle_ps(v, v, 0xaa), _other.rows[2].getAsRegister()));
            v = _mm_add_ps(v, _mm_mul_ps(_mm_shuffle_ps(v, v, 0xff), _other.rows[3].getAsRegister()));

            return vectorSIMDf{v};
        };

        matrix4SIMD r;
        for (size_t i = 0u; i < 4u; ++i)
            r.rows[i] = calcRow(rows[i]);

        *this = r;
    }
    inline matrix4SIMD operator*(const matrix4SIMD& _other) const
    {
        matrix4SIMD r{*this};
        return r *= _other;
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
        return (*this * getTransposed()).isIdentity();
    }
    inline bool isOrthogonal(float _tolerance) const
    {
        return (*this * getTransposed()).isIdentity(_tolerance);
    }

    inline matrix4SIMD& setScale(const core::vectorSIMDf& _scale)
    {
        const vectorSIMDf mask0001 = BUILD_MASKF(0, 0, 0, 1);

        rows[0] = (_scale & BUILD_MASKF(1, 0, 0, 0)) | (rows[0] & mask0001);
        rows[1] = (_scale & BUILD_MASKF(0, 1, 0, 0)) | (rows[1] & mask0001);
        rows[2] = (_scale & BUILD_MASKF(0, 0, 1, 0)) | (rows[2] & mask0001);
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

        vectorSIMDf mask1110 = BUILD_MASKF(1, 1, 1, 0);
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

    //! Modifies only upper-left 3x3.
    inline matrix4SIMD& setRotation(const quaternion& _quat)
    {
        const __m128 mask0001 = BUILD_MASKF(0, 0, 0, 1);
        const __m128 mask1110 = BUILD_MASKF(1, 1, 1, 0);

        const vectorSIMDf& quat = reinterpret_cast<const vectorSIMDf&>(_quat);
        rows[0] = ((quat.yyyy() * ((quat.yxwx() & mask1110) * vectorSIMDf(2.f))) + (quat.zzzz() * (quat.zwxx() & mask1110) * vectorSIMDf(2.f, -2.f, 2.f, 0.f))) | (rows[0] & mask0001);
        rows[0].x = 1.f - rows[0].x;

        rows[1] = ((quat.zzzz() * ((quat.wzyx() & mask1110) * vectorSIMDf(2.f))) + (quat.xxxx() * (quat.yxwx() & mask1110) * vectorSIMDf(2.f, 2.f, -2.f, 0.f))) | (rows[1] & mask0001);
        rows[1].y = 1.f - rows[1].y;

        rows[2] = ((quat.xxxx() * ((quat.zwxx() & mask1110) * vectorSIMDf(2.f))) + (quat.yyyy() * (quat.wzyx() & mask1110) * vectorSIMDf(-2.f, 2.f, 2.f, 0.f))) | (rows[2] & mask0001);
        rows[2].z = 1.f - rows[2].z;

        return *this;
    }

    //! W component remains unmodified.
    inline void rotateVect(vectorSIMDf& _out, const vectorSIMDf& _in) const
    {
        matrix4SIMD cp{*this};
        core::transpose4(cp.rows);
        __m128 m1110 = BUILD_MASKF(1, 1, 1, 0);
        for (size_t i = 0u; i < 3u; ++i)
            cp.rows[i] &= m1110;

        _out.X = _in.dotProductAsFloat(cp.rows[0]);
        _out.Y = _in.dotProductAsFloat(cp.rows[1]);
        _out.Z = _in.dotProductAsFloat(cp.rows[2]);
    }
    //! W component remains unmodified.
    inline void rotateVect(vectorSIMDf& _vector) const
    {
        const vectorSIMDf in = _vector;
        rotateVect(_vector, in);
    }
    //! An alternate transform vector method, writing into an array of 3 floats
    inline void rotateVect(float* _out, const vectorSIMDf& _in) const
    {
        vectorSIMDf out;
        rotateVect(out, _in);
        memcpy(_out, out.pointer, 3*4);
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
    inline void transformVect(float* _out, const vectorSIMDf& _in) const
    {
        vectorSIMDf outv;
        transformVect(outv, _in);
        _mm_storeu_ps(_out, outv.getAsRegister());
    }

    inline void translateVect(vectorSIMDf& _vect) const
    {
        _vect += getTranslation();
    }

#define BUILD_SHUFFLE_MASK(_x_, _y_, _z_, _w_) (_x | (_y<<2) | (z<<4) | (w<<6))
    inline matrix4SIMD& buildProjectionMatrixPerspectiveFovRH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
    {
        const double h = core::reciprocal(tan(fieldOfViewRadians*0.5));
        _IRR_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
        const float w = h / aspectRatio;

        _IRR_DEBUG_BREAK_IF(zNear == zFar); //division by zero

        rows[0] = vectorSIMDf(w, 0.f, 0.f, 0.f);
        rows[1] = vectorSIMDf(0.f, (float)h, 0.f, 0.f);
        rows[2] = vectorSIMDf(0.f, 0.f, zFar + zNear/(zNear - zFar), 0.f);
        rows[3] = vectorSIMDf(0.f, 0.f, 2.f*zFar*zNear/(zNear-zFar), 0.f);

        return *this;
    }
    inline matrix4SIMD& buildProjectionMatrixPerspectiveFovLH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
    {
        const double h = core::reciprocal(tan(fieldOfViewRadians*0.5));
        _IRR_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
        const float w = h / aspectRatio;

        _IRR_DEBUG_BREAK_IF(zNear == zFar); //division by zero

        rows[0] = vectorSIMDf(w, 0.f, 0.f, 0.f);
        rows[1] = vectorSIMDf(0.f, (float)h, 0.f, 0.f);
        rows[2] = vectorSIMDf(0.f, 0.f, zFar + zNear / (zFar - zNear), 0.f);
        rows[3] = vectorSIMDf(0.f, 0.f, -2.f*zFar*zNear / (zFar - zNear), 0.f);

        return *this;
    }

    inline matrix4SIMD& buildProjectionMatrixPerspectiveFovInfinityLH(float fieldOfViewRadians, float aspectRatio, float zNear, float epsilon)
    {
        const double h = core::reciprocal(tan(fieldOfViewRadians*0.5));
        _IRR_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
        const float w = h / aspectRatio;

        rows[0] = vectorSIMDf(w, 0.f, 0.f, 0.f);
        rows[1] = vectorSIMDf(0.f, (float)h, 0.f, 0.f);
        rows[2] = vectorSIMDf(0.f, 0.f, 1.f - epsilon, 0.f);
        rows[3] = vectorSIMDf(0.f, 0.f, zNear*(epsilon - 1.f), 0.f);

        return *this;
    }

    inline matrix4SIMD& buildProjectionMatrixOrthoLH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
    {
        _IRR_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(zNear == zFar); //division by zero

        rows[0] = vectorSIMDf(2.f/widthOfViewVolume, 0.f, 0.f, 0.f);
        rows[1] = vectorSIMDf(0.f, 2.f/heightOfViewVolume, 0.f, 0.f);
        rows[2] = vectorSIMDf(0.f, 0.f, 1.f/(zFar-zNear), 0.f);
        rows[3] = vectorSIMDf(0.f, 0.f, zNear/(zNear-zFar), 1.f);
    }
    inline matrix4SIMD& buildProjectionMatrixOrthoRH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
    {
        _IRR_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(zNear == zFar); //division by zero

        rows[0] = vectorSIMDf(2.f/widthOfViewVolume, 0.f, 0.f, 0.f);
        rows[1] = vectorSIMDf(0.f, 2.f/heightOfViewVolume, 0.f, 0.f);
        rows[2] = vectorSIMDf(0.f, 0.f, 1.f/(zNear-zFar), 0.f);
        rows[3] = vectorSIMDf(0.f, 0.f, zNear/(zNear-zFar), 1.f);
    }

    inline matrix4SIMD& buildProjectionMatrixPerspectiveRH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
    {
        _IRR_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(zNear == zFar); //division by zero

        rows[0] = vectorSIMDf(2.f*zNear/widthOfViewVolume, 0.f, 0.f, 0.f);
        rows[1] = vectorSIMDf(0.f, 2.f*zNear/heightOfViewVolume, 0.f, 0.f);
        rows[2] = vectorSIMDf(0.f, 0.f, zFar/(zNear-zFar), -1.f);
        rows[3] = vectorSIMDf(0.f, 0.f, zNear*zFar/(zNear-zFar), 0.f);

        return *this;
    }
    inline matrix4SIMD& buildProjectionMatrixPerspectiveLH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
    {
        _IRR_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
        _IRR_DEBUG_BREAK_IF(zNear == zFar); //division by zero

        rows[0] = vectorSIMDf(2.f*zNear/widthOfViewVolume, 0.f, 0.f, 0.f);
        rows[1] = vectorSIMDf(0.f, 2.f*zNear/heightOfViewVolume, 0.f, 0.f);
        rows[2] = vectorSIMDf(0.f, 0.f, zFar/(zFar-zNear), -1.f);
        rows[3] = vectorSIMDf(0.f, 0.f, zNear*zFar/(zNear-zFar), 0.f);

        return *this;
    }

    inline matrix4SIMD& buildShadowMatrix(vectorSIMDf _light, const core::plane3df& _plane, float _point)
    {
        const vectorSIMDf mask1110 = BUILD_MASKF(1, 1, 1, 0);
        vectorSIMDf normal = vectorSIMDf(&_plane.Normal.X) & mask1110;
        normal = core::normalize(normal);
        const vectorSIMDf d = normal.dotProduct(_light);
        normal.w = _plane.D;

        _light.w = _point;

        rows[0] = (-normal.xxxx() * _light) + (d & BUILD_MASKF(1, 0, 0, 0));
        rows[1] = (-normal.yyyy() * _light) + (d & BUILD_MASKF(0, 1, 0, 0));
        rows[2] = (-normal.zzzz() * _light) + (d & BUILD_MASKF(0, 0, 1, 0));
        rows[3] = (-normal.wwww() * _light) + (d & BUILD_MASKF(0, 0, 0, 1));

        return *this;
    }

    inline static matrix4SIMD buildCameraLookAtMatrixLH(
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
    inline static matrix4SIMD buildCameraLookAtMatrixRH(
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

    inline matrix4SIMD interpolate(const matrix4SIMD& _b, float _x) const
    {
        matrix4SIMD m;
        for (size_t i = 0u; i < 4u; ++i)
            m.rows[i] = core::mix(rows[i], _b.rows[i], vectorSIMDf(_x));

        return m;
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

    inline matrix4SIMD& buildRotateFromTo(const core::vectorSIMDf& from, const core::vectorSIMDf& to)
	{
		// unit vectors
		const core::vectorSIMDf f = core::normalize(from);
		const core::vectorSIMDf t = core::normalize(to);

		// axis multiplication by sin
		const core::vectorSIMDf vs(t.crossProduct(f));

		// axis of rotation
		const core::vectorSIMDf v = core::normalize(vs);

		// cosinus angle
		const core::vectorSIMDf ca = f.dotProduct(t);

		const core::vectorSIMDf vt(v * (core::vectorSIMDf(1.f) - ca));
		const core::vectorSIMDf wt = vt * v.yzxx();
		const core::vectorSIMDf vtuppca = vt * v + ca;

		core::vectorSIMDf& row0 = rows[0];
		core::vectorSIMDf& row1 = rows[1];
		core::vectorSIMDf& row2 = rows[2];
        core::vectorSIMDf& row3 = rows[3];

		const core::vectorSIMDf mask0001 = BUILD_MASKF(0, 0, 0, 1);
		row0 = (row0 & mask0001) + (vtuppca & BUILD_MASKF(1, 0, 0, 0));
		row1 = (row1 & mask0001) + (vtuppca & BUILD_MASKF(0, 1, 0, 0));
		row2 = (row2 & mask0001) + (vtuppca & BUILD_MASKF(0, 0, 1, 0));

		row0 += (wt.xxzx() + vs.xzyx()*core::vectorSIMDf(1.f, 1.f, -1.f, 1.f)) & BUILD_MASKF(0, 1, 1, 0);
		row1 += (wt.xxyx() + vs.zxxx()*core::vectorSIMDf(-1.f, 1.f, 1.f, 1.f)) & BUILD_MASKF(1, 0, 1, 0);
		row2 += (wt.zyxx() + vs.yxxx()*core::vectorSIMDf(1.f, -1.f, 1.f, 1.f)) & BUILD_MASKF(1, 1, 0, 0);
        row3 = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

		return *this;
	}

    inline void setRotationCenter(const core::vectorSIMDf& _center, const core::vectorSIMDf& _translation)
    {
        core::vectorSIMDf r0 = rows[0] * _center;
        core::vectorSIMDf r1 = rows[1] * _center;
        core::vectorSIMDf r2 = rows[2] * _center;
        core::vectorSIMDf r3(0.f, 0.f, 0.f, 1.f);

        __m128 col3 = _mm_hadd_ps(_mm_hadd_ps(r0.getAsRegister(), r1.getAsRegister()), _mm_hadd_ps(r2.getAsRegister(), r3.getAsRegister()));
        const vectorSIMDf vcol3 = _center - _translation - col3;

        setTranslation(vcol3);
    }

    inline void buildAxisAlignedBillboard(
        const core::vectorSIMDf& _camPos,
        const core::vectorSIMDf& _center,
        const core::vectorSIMDf& _translation,
        const core::vectorSIMDf& _axis,
        const core::vectorSIMDf& _from)
    {
        // axis of rotation
        const core::vectorSIMDf up = core::normalize(_axis);
        const core::vectorSIMDf forward = core::normalize(_camPos - _center);
        const core::vectorSIMDf right = core::normalize(up.crossProduct(forward));

        // correct look vector
        const core::vectorSIMDf look = right.crossProduct(up);

        // rotate from to
        // axis multiplication by sin
        const core::vectorSIMDf vs = look.crossProduct(_from);

        // cosinus angle
        const core::vectorSIMDf ca = _from.dotProduct(look);

        const core::vectorSIMDf vt(up * (core::vectorSIMDf(1.f) - ca));
        const core::vectorSIMDf wt = vt * up.yzxx();
        const core::vectorSIMDf vtuppca = vt * up + ca;

        core::vectorSIMDf& row0 = rows[0];
        core::vectorSIMDf& row1 = rows[1];
        core::vectorSIMDf& row2 = rows[2];

        row0 = vtuppca & BUILD_MASKF(1, 0, 0, 0);
        row1 = vtuppca & BUILD_MASKF(0, 1, 0, 0);
        row2 = vtuppca & BUILD_MASKF(0, 0, 1, 0);

        row0 += (wt.xxzx() + vs.xzyx()*core::vectorSIMDf(1.f, 1.f, -1.f, 1.f)) & BUILD_MASKF(0, 1, 1, 0);
        row1 += (wt.xxyx() + vs.zxxx()*core::vectorSIMDf(-1.f, 1.f, 1.f, 1.f)) & BUILD_MASKF(1, 0, 1, 0);
        row2 += (wt.zyxx() + vs.yxxx()*core::vectorSIMDf(1.f, -1.f, 1.f, 1.f)) & BUILD_MASKF(1, 1, 0, 0);

        setRotationCenter(_center, _translation);
    }

    inline matrix4SIMD& buildNDCToDCMatrix(const core::rect<int32_t>& _viewport, float _zScale)
    {
        const float scaleX = (float(_viewport.getWidth()) - 0.75f) * 0.5f;
        const float scaleY = -(float(_viewport.getHeight()) - 0.75f) * 0.5f;

        const float dx = -0.5f + ((_viewport.UpperLeftCorner.X + _viewport.LowerRightCorner.X) * 0.5f);
        const float dy = -0.5f + ((_viewport.UpperLeftCorner.Y + _viewport.LowerRightCorner.Y) * 0.5f);

        *this = matrix4SIMD();
        rows[3] = vectorSIMDf(dx, dy, 0.f, 1.f);
        return setScale(vectorSIMDf(scaleX, scaleY, _zScale, 1.f));
    }

#define BUILD_XORMASKF(_x_, _y_, _z_, _w_) _mm_castsi128_ps(_mm_setr_epi32(_x_*0x80000000, _y_*0x80000000, _z_*0x80000000, _w_*0x80000000))
    inline matrix4SIMD& buildTextureTransform(
        float _rotateRad,
        const core::vector2df& _rotateCenter,
        const core::vector2df& _translate,
        const core::vector2df& _scale)
    {
        const vectorSIMDf mask1100 = BUILD_MASKF(1, 1, 0, 0);

        const vectorSIMDf scale = vectorSIMDf(&_scale.X) & mask1100;
        const vectorSIMDf rotateCenter = vectorSIMDf(&_rotateCenter.X) & mask1100;
        const vectorSIMDf translation = vectorSIMDf(&_translate.X) & mask1100;

        vectorSIMDf cossin(cosf(_rotateRad), sinf(_rotateRad), 0.f, 0.f);
        
        rows[0] = cossin * scale;
        rows[1] = (cossin ^ BUILD_XORMASKF(0, 1, 0, 0)).yxww() * scale;
        rows[2] = cossin * scale * rotateCenter.xxww() + (cossin.yxww() ^ BUILD_XORMASKF(1, 0, 0, 0)) * rotateCenter.yyww() + translation;
        rows[3] = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

        return *this;
    }

    inline matrix4SIMD& setTextureRotationCenter(float _rotateRad)
    {
        vectorSIMDf cossin(cosf(_rotateRad), sinf(_rotateRad), 0.f, 0.f);
        const vectorSIMDf mask0011 = BUILD_MASKF(0, 0, 1, 1);

        rows[0] = (rows[0] & mask0011) | cossin;
        rows[1] = (rows[1] & mask0011) | (cossin.yxww() ^ BUILD_XORMASKF(1, 0, 0, 0));
        rows[2] = (rows[2] & mask0011) | vectorSIMDf(0.5f * (cossin.y - cossin.x) + 0.5f, 0.5f * (cossin.x + cossin.y) + 0.5f, 0.f, 0.f);

        return *this;
    }

    inline matrix4SIMD& setTextureTranslate(float _x, float _y)
    {
        rows[2] = (rows[2] & BUILD_MASKF(0, 0, 1, 1)) | vectorSIMDf(_x, _y, 0.f, 0.f);

        return *this;
    }

    inline matrix4SIMD& setTextureScale(float _sx, float _sy)
    {
        rows[0].x = _sx;
        rows[1].y = _sy;

        return *this;
    }

    inline matrix4SIMD& setTextureScaleCenter(float _sx, float _sy)
    {
        rows[2] = (rows[2] & BUILD_MASKF(0, 0, 1, 1)) | vectorSIMDf(0.5f - 0.5f*_sx, 0.5f - 0.5f*_sy, 0.f, 0.f);
        return setTextureScale(_sx, _sy);
    }

    inline bool equals(const matrix4SIMD& _other, float _tolerance) const
    {
        for (size_t i = 0u; i < 4u; ++i)
            if (!core::equals(rows[i], _other.rows[i], _tolerance).all())
                return false;
        return true;
    }

#undef BUILD_MASKF
#undef BUILD_XORMASKF
}
#ifndef _IRR_WINDOWS_
    __attribute__((__aligned__(SIMD_ALIGNMENT)))
#endif
;

inline matrix4SIMD operator*(float _scalar, const matrix4SIMD& _mtx)
{
    return _mtx * _scalar;
}

}} // irr::core

#endif // __IRR_MATRIX4SIMD_H_INCLUDED__
