// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_CORE_PLANE_3D_H_INCLUDED__
#define __NBL_CORE_PLANE_3D_H_INCLUDED__

#include "matrix3x4SIMD.h"

namespace nbl
{
namespace core
{

//! Enumeration for intersection relations of 3d objects
enum EIntersectionRelation3D
{
	ISREL3D_FRONT = 0,
	ISREL3D_BACK,
	ISREL3D_PLANAR,
	ISREL3D_SPANNING,
	ISREL3D_CLIPPED
};

//! Template plane class with some intersection testing methods.
/*
	It has to be ensured, that the normal is always normalized. The constructors
    and setters of this class will not ensure this automatically. So any normal
    passed in has to be normalized in advance. No change to the normal will be
    made by any of the class methods.
*/
class plane3dSIMDf : private vectorSIMDf
{
	public:
	    inline plane3dSIMDf() : vectorSIMDf() {}
		//
		inline plane3dSIMDf(const vectorSIMDf& planeEq) : vectorSIMDf(planeEq) {}
		//
		inline plane3dSIMDf(const vector3df_SIMD& MPoint, const vector3df_SIMD& Normal_in) : vectorSIMDf(Normal_in)
		{
		    recalculateD(MPoint);
        }
        //
		inline plane3dSIMDf(const vector3df_SIMD& point1, const vector3df_SIMD& point2, const vector3df_SIMD& point3)
		{
		    setPlane(point1, point2, point3);
        }

		// operators
		inline bool operator==(const plane3dSIMDf& other) const { return (vectorSIMDf::operator==(other)).all();}

		inline bool operator!=(const plane3dSIMDf& other) const { return (vectorSIMDf::operator!=(other)).any();}

		// functions
		inline void setPlane(const vectorSIMDf& planeEq)
		{
			*static_cast<vectorSIMDf*>(this) = planeEq;
		}
		//
		inline void setPlane(const vector3df_SIMD& point, const vector3df_SIMD& nvector)
		{
			*static_cast<vectorSIMDf*>(this) = nvector;
			recalculateD(point);
		}
        // creates the plane from 3 memberpoints
		inline void setPlane(const vector3df_SIMD& point1, const vector3df_SIMD& point2, const vector3df_SIMD& point3, bool shouldNormalize=true)
		{
			auto& Normal = *static_cast<vectorSIMDf*>(this);
			Normal = cross(point2 - point1,point3 - point1);
			if (shouldNormalize)
                Normal = normalize(Normal);

			recalculateD(point1);
		}

		inline vectorSIMDf& getPlaneEq() {return *this;}
		inline const vectorSIMDf& getPlaneEq() const {return *this;}

		inline float& getDistance() {return w;}
		inline float getDistance() const {return w;}

		inline vector3df_SIMD getNormal() const
		{
		    vectorSIMDf normal = getPlaneEq();
		    normal.makeSafe3D();
		    return normal;
		}
		inline void setNormal(vector3df_SIMD normal)
		{
            #define BUILD_MASKF(_x_, _y_, _z_, _w_) _mm_setr_epi32(_x_*0xffffffff, _y_*0xffffffff, _z_*0xffffffff, _w_*0xffffffff)
		    getPlaneEq() &= BUILD_MASKF(0,0,0,1);
		    getPlaneEq() |= _mm_castps_si128((normal&BUILD_MASKF(1,1,1,0)).getAsRegister());
		}

		//! Classifies the relation of a point to this plane.
		/* 
			\param point Point to classify its relation.
			\return ISREL3D_FRONT if the point is in front of the plane,
			ISREL3D_BACK if the point is behind of the plane, and
			ISREL3D_PLANAR if the point is within the plane. 
		*/
		EIntersectionRelation3D classifyPointRelation(const vector3df& point) const
		{
			static constexpr float ROUNDING_ERROR_f32 = 0.000001f;

			const auto& normal = getNormal();
			const float d = dot(vector3df(normal.x, normal.y, normal.z), point).X + w;

			if (d < -ROUNDING_ERROR_f32)
				return ISREL3D_BACK;

			if (d > ROUNDING_ERROR_f32)
				return ISREL3D_FRONT;

			return ISREL3D_PLANAR;
		}

		//! Recalculates the distance from origin by applying a new member point to the plane.
		inline void recalculateD(const vector3df_SIMD& MPoint)
		{
			w = -dot(getNormal(),MPoint).x;
		}

		//! Gets a member point of the plane.
		inline vector3df_SIMD getMemberPoint() const
		{
		    vectorSIMDf pt(*this);
		    pt *= pt.wwww();
		    pt.makeSafe3D();
		    return pt;
		}

		//!
        static inline plane3dSIMDf transform(const plane3dSIMDf& _in, const matrix3x4SIMD& _mat)
        {
            matrix3x4SIMD inv;
            _mat.getInverse(inv);

            vectorSIMDf normal(_in.getNormal());
            // transform by inverse transpose
            return plane3dSIMDf(inv.rows[0]*normal.xxxx()+inv.rows[1]*normal.yyyy()+inv.rows[2]*normal.zzzz()+(normal.wwww()&BUILD_MASKF(0,0,0,1)));
		    #undef BUILD_MASKF
        }

		//! Tests if there is an intersection with the other plane
		/** \return True if there is a intersection. */
		inline bool existsIntersection(const plane3dSIMDf& other) const
		{
			return length(cross(other.getNormal(),getNormal())).x > 0.f;
		}

		//! Get an intersection with a 3d line.
		/** \param lineVect Vector of the line to intersect with. HAS TO HAVE W component EQUAL TO 0.0 !!!
		\param linePoint Point of the line to intersect with. HAS TO HAVE W component EQUAL TO 1.0 !!!
		\param outIntersection Place to store the intersection point, if there is one. Will have W component equal to 1.
		\return True if there was an intersection, false if there was not.
		*/
		bool getIntersectionWithLine(const vectorSIMDf& linePoint, const vector3df_SIMD& lineVect, vector3df_SIMD& outIntersection) const
		{
			vectorSIMDf t2 = dot(*static_cast<const vectorSIMDf*>(this),lineVect);

			if (t2.x == 0.f)
				return false;

			vectorSIMDf t = -dot(*static_cast<const vectorSIMDf*>(this),linePoint)/t2;
			outIntersection = linePoint + lineVect * t;
			return true;
		}

		//! Get percentage of line between two points where an intersection with this plane happens.
		/*
			Only useful if known that there is an intersection.
			\param linePoint1 Point1 of the line to intersect with.
			\param linePoint2 Point2 of the line to intersect with.
			\return Where on a line between two points an intersection with this plane happened.
			For example, 0.5 is returned if the intersection happened exactly in the middle of the two points.
		*/
		float getKnownIntersectionWithLine(const vector3df& linePoint1, const vector3df& linePoint2) const
		{
			vector3df vect = linePoint2 - linePoint1;
			const auto& normal = getNormal();
			float t2 = dot(vector3df(normal.x, normal.y, normal.z), vect).X;
			return static_cast<float>(-((dot(vector3df(normal.x, normal.y, normal.z), linePoint1).X + W) / t2));
		}

		//! Intersects this plane with another.
		/** \param other Other plane to intersect with.
		\param outLinePoint Base point of intersection line.
		\param outLineVect Vector of intersection.
		\return True if there is a intersection, false if not. */
		bool getIntersectionWithPlane(const plane3dSIMDf& other,
				vector3df_SIMD& outLinePoint,
				vector3df_SIMD& outLineVect) const
		{
		    auto n1(getNormal());
		    auto n2(other.getNormal());
			outLineVect = cross(n1,n2);
			if ((outLineVect==vectorSIMDf(0.f)).all())
                return false;

			outLinePoint = cross(n1*other.wwww()-n2*this->wwww(),outLineVect)/dot(outLineVect,outLineVect);
			return true;
		}

		//! Get the intersection point with two other planes if there is one.
		bool getIntersectionWithPlanes(const plane3dSIMDf& o1, const plane3dSIMDf& o2, vectorSIMDf& outPoint) const
		{
			vectorSIMDf linePoint, lineVect;
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


