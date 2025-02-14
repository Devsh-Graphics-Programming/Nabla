// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_BUILTIN_HLSL_PLANE_HLSL_INCLUDED__
#define __NBL_BUILTIN_HLSL_PLANE_HLSL_INCLUDED__

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct plane;

//! Template plane class with some intersection testing methods.
/** It has to be ensured, that the normal is always normalized. The constructors
    and setters of this class will not ensure this automatically. So any normal
    passed in has to be normalized in advance. No change to the normal will be
    made by any of the class methods.
*/
template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct plane<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
	using this_t = plane<T>;
	vector<T, 4> data;

	NBL_CONSTEXPR_STATIC_INLINE this_t create()
	{
		this_t output;
		output.data = vector<T, 4>(0);
		return output;
	}
	//
	NBL_CONSTEXPR_STATIC_INLINE this_t create(NBL_CONST_REF_ARG(vector<T, 4>) planeEq)
	{
		this_t output;
		output.data = planeEq;
		return output;
	}
	//
	NBL_CONSTEXPR_STATIC_INLINE this_t create(NBL_CONST_REF_ARG(vector<T, 3>) MPoint, NBL_CONST_REF_ARG(vector<T, 3>) Normal_in)
	{
		this_t output;
		output.data = vector<T, 4>(Normal_in.x, Normal_in.y, Normal_in.z, 0);
	    output.recalculateD(MPoint);
		return output;
	}
	//
	NBL_CONSTEXPR_STATIC_INLINE this_t create(NBL_CONST_REF_ARG(vector<T, 3>) point1, NBL_CONST_REF_ARG(vector<T, 3>) point2, NBL_CONST_REF_ARG(vector<T, 3>) point3)
	{
		this_t output;
		output.setPlane(point1, point2, point3);
		return output;
    }

	// operators
	inline bool operator==(NBL_CONST_REF_ARG(this_t) other) NBL_CONST_MEMBER_FUNC { return create(data) == other; }
	inline bool operator!=(NBL_CONST_REF_ARG(this_t) other) NBL_CONST_MEMBER_FUNC { return create(data) != other; }

	// functions
	inline void setPlane(NBL_CONST_REF_ARG(vector<T, 4>) planeEq)
	{
		data = planeEq;
	}
	//
	inline void setPlane(NBL_CONST_REF_ARG(vector<T, 3>) _point, NBL_CONST_REF_ARG(vector<T, 3>) nvector)
	{
		data = vector<T, 4>(nvector, 0);
		recalculateD(_point);
	}
	// creates the plane from 3 memberpoints
	inline void setPlane(NBL_CONST_REF_ARG(vector<T, 3>) point1, NBL_CONST_REF_ARG(vector<T, 3>) point2, NBL_CONST_REF_ARG(vector<T, 3>) point3, bool shouldNormalize = true)
	{
		vector<T, 3> Normal = hlsl::cross(point2 - point1,point3 - point1);
		if (shouldNormalize)
               Normal = normalize(Normal);
		data = vector<T, 4>(Normal, 0);

		recalculateD(point1);
	}

	inline vector<T, 4> getPlaneEq() NBL_CONST_MEMBER_FUNC {return data;}

	inline T getDistance() NBL_CONST_MEMBER_FUNC { return data.w; }
	inline void setDistance(T distance) { data.w = distance; }

	inline vector<T, 3> getNormal() NBL_CONST_MEMBER_FUNC
	{
	    return vector<T, 3>(data.x, data.y, data.z);
	}
	inline void setNormal(NBL_CONST_REF_ARG(vector<T, 3>) normal)
	{
		data.x = normal.x;
		data.y = normal.y;
		data.z = normal.z;
	}

	//! Recalculates the distance from origin by applying a new member point to the plane.
	inline void recalculateD(NBL_CONST_REF_ARG(vector<T, 3>) MPoint)
	{
		data.w = -hlsl::dot(getNormal(),MPoint);
	}

	//! Gets a member point of the plane.
	inline vector<T, 3> getMemberPoint() NBL_CONST_MEMBER_FUNC
	{
	    return getNormal() * getDistance();
	}

	//!
    static inline this_t transform(NBL_CONST_REF_ARG(this_t) _in, NBL_CONST_REF_ARG(matrix<T, 4, 4>) _mat)
    {
		matrix<T, 4, 4> inv = hlsl::inverse(_mat);
        // transform by inverse transpose
		this_t output = create(inv[0] * _in.data.x + inv[1] * _in.data.y + inv[2] * _in.data.z);
		output.data.w += _in.data.w;

        return output;
    }

	//! Tests if there is an intersection with the other plane
	/** \return True if there is a intersection. */
	inline bool existsIntersection(NBL_CONST_REF_ARG(this_t) other) NBL_CONST_MEMBER_FUNC
	{
		return hlsl::length(hlsl::cross(other.getNormal(),getNormal())) > 0.f;
	}

	//! Get an intersection with a 3d line.
	/** \param lineVect Vector of the line to intersect with.
	\param linePoint Point of the line to intersect with.
	\param outIntersection Place to store the intersection point, if there is one. Will have W component equal to 1.
	\return True if there was an intersection, false if there was not.
	*/
	bool getIntersectionWithLine(NBL_CONST_REF_ARG(vector<T, 3>) linePoint, NBL_CONST_REF_ARG(vector<T, 3>) lineVect, NBL_REF_ARG(vector<T, 3>) outIntersection) NBL_CONST_MEMBER_FUNC
	{
		T t2 = hlsl::dot(data,vector<T, 4>(lineVect, 0));

		if (t2 == 0.f)
			return false;

		T t = -dot(data,vector<T, 4>(linePoint, 1))/t2;
		outIntersection = linePoint + lineVect * t;
		return true;
	}

	//! Intersects this plane with another.
	/** \param other Other plane to intersect with.
	\param outLinePoint Base point of intersection line.
	\param outLineVect Vector of intersection.
	\return True if there is a intersection, false if not. */
	bool getIntersectionWithPlane(NBL_CONST_REF_ARG(this_t) other,
			NBL_REF_ARG(vector<T, 3>) outLinePoint,
			NBL_REF_ARG(vector<T, 3>) outLineVect) NBL_CONST_MEMBER_FUNC
	{
		vector<T, 3> n1 = getNormal();
		vector<T, 3> n2 = other.getNormal();
		outLineVect = hlsl::cross(n1,n2);
		if (outLineVect.x == 0 && outLineVect.y == 0 && outLineVect.z == 0)
               return false;

		outLinePoint = hlsl::cross(n1*other.data.w-n2*data.w,outLineVect)/dot(outLineVect,outLineVect);
		return true;
	}

	//! Get the intersection point with two other planes if there is one.
	bool getIntersectionWithPlanes(NBL_CONST_REF_ARG(plane) o1, NBL_CONST_REF_ARG(plane) o2, NBL_CONST_REF_ARG(vector<T, 4>) outPoint) NBL_CONST_MEMBER_FUNC
	{
		vector<T, 3> linePoint, lineVect;
		if (getIntersectionWithPlane(o1, linePoint, lineVect))
        {
            linePoint.w = 1.f;
			return o2.getIntersectionWithLine(linePoint, lineVect, outPoint);
        }

		return false;
	}
};

} // end namespace core
} // end namespace nbl

#endif


