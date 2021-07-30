// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_S_VIEW_FRUSTUM_H_INCLUDED__
#define __NBL_S_VIEW_FRUSTUM_H_INCLUDED__

#include "nbl/core/core.h"
#include "matrixutil.h"

namespace nbl
{
namespace scene
{

	//! Defines the view frustum. That's the space visible by the camera.
	/** The view frustum is enclosed by 6 planes. These six planes share
	eight points. A bounding box around these eight points is also stored in
	this structure.
	*/
	struct SViewFrustum
	{
		enum VFPLANES
		{
			//! Far plane of the frustum. That is the plane farest away from the eye.
			VF_FAR_PLANE = 0,
			//! Near plane of the frustum. That is the plane nearest to the eye.
			VF_NEAR_PLANE,
			//! Left plane of the frustum.
			VF_LEFT_PLANE,
			//! Right plane of the frustum.
			VF_RIGHT_PLANE,
			//! Bottom plane of the frustum.
			VF_BOTTOM_PLANE,
			//! Top plane of the frustum.
			VF_TOP_PLANE,

			//! Amount of planes enclosing the view frustum. Should be 6.
			VF_PLANE_COUNT
		};


		//! Default Constructor.
		SViewFrustum() {}

		//! Copy Constructor.
		SViewFrustum(const SViewFrustum& other) = default;

		//! This constructor creates a view frustum based on a projection and/or view matrix.
		/** @param mat Source matrix. */
		inline explicit SViewFrustum(const core::matrix4SIMD& mat) { setFrom ( mat );}

		//! Modifies frustum as if it was constructed with a matrix.
		/** @param mat Source matrix.
		@see @ref SViewFrustum(const core::matrix4SIMD&)
		*/
		inline void setFrom(const core::matrix4SIMD& mat);

		//! @returns the point which is on the far left upper corner inside the the view frustum.
		core::vector3df_SIMD getFarLeftUp() const;

		//! @returns the point which is on the far left bottom corner inside the the view frustum.
		core::vector3df_SIMD getFarLeftDown() const;

		//! @returns the point which is on the far right top corner inside the the view frustum.
		core::vector3df_SIMD getFarRightUp() const;

		//! @returns the point which is on the far right bottom corner inside the the view frustum.
		core::vector3df_SIMD getFarRightDown() const;

		//! @returns the point which is on the near left upper corner inside the the view frustum.
		core::vector3df_SIMD getNearLeftUp() const;

		//! @returns the point which is on the near left bottom corner inside the the view frustum.
		core::vector3df_SIMD getNearLeftDown() const;

		//! @returns the point which is on the near right top corner inside the the view frustum.
		core::vector3df_SIMD getNearRightUp() const;

		//! @returns the point which is on the near right bottom corner inside the the view frustum.
		core::vector3df_SIMD getNearRightDown() const;

		//! Recalculates the bounding box member based on the planes.
        inline void recalculateBoundingBox()
        {
            boundingBox.reset(getNearLeftUp().getAsVector3df());
            boundingBox.addInternalPoint(getNearRightUp().getAsVector3df());
            boundingBox.addInternalPoint(getNearLeftDown().getAsVector3df());
            boundingBox.addInternalPoint(getNearRightDown().getAsVector3df());

            boundingBox.addInternalPoint(getFarLeftUp().getAsVector3df());
            boundingBox.addInternalPoint(getFarRightUp().getAsVector3df());
            boundingBox.addInternalPoint(getFarLeftDown().getAsVector3df());
            boundingBox.addInternalPoint(getFarRightDown().getAsVector3df());
        }

        inline bool intersectsAABB(const core::aabbox3df& box) const
        {
            core::vectorSIMDf inMinPt(&box.MinEdge.X);
            // TODO: after change to SSE aabbox3df
            core::vectorSIMDf inMaxPt(box.MaxEdge.X,box.MaxEdge.Y,box.MaxEdge.Z,1.f);
            // TODO: after change to aabbox3dSIMDf we will just be able to assume these things
            inMinPt.w = 1.f;

            const auto zero = core::vectorSIMDf(0.f);
            auto getMaxCoord = [&](const core::plane3dSIMDf& plane)
            {
                const auto& planeAsVec4 = reinterpret_cast<const core::vectorSIMDf&>(plane);
                auto retval = core::mix(inMinPt,inMaxPt,planeAsVec4>zero);
                retval *= planeAsVec4;
                return retval; // retval.w ==plane.w
            };
            auto doHadd = [](const core::vectorSIMDf& a, const core::vectorSIMDf& b)
            {
                return core::vectorSIMDf(_mm_hadd_ps(a.getAsRegister(),b.getAsRegister()));
            };

            // near plane
            auto zPlaneDistances = doHadd(doHadd(getMaxCoord(planes[VF_NEAR_PLANE]),getMaxCoord(planes[VF_FAR_PLANE])),core::vectorSIMDf(123456789.f));
            if ((zPlaneDistances<zero).any())
                return false;

            core::vectorSIMDf xyPlaneDistances;
            {
                auto half0 = doHadd(getMaxCoord(planes[VF_LEFT_PLANE]),getMaxCoord(planes[VF_RIGHT_PLANE]));
                auto half1 = doHadd(getMaxCoord(planes[VF_TOP_PLANE]),getMaxCoord(planes[VF_BOTTOM_PLANE]));
                xyPlaneDistances = doHadd(half0,half1);
            }
            // NDC xy -1 and +1 planes
            if ((xyPlaneDistances<zero).any())
                return false;

            return true;
        }

		//! Transforms the frustum by the matrix
		/** @param mat: Matrix by which the view frustum is transformed.*/
		void transform(const core::matrix3x4SIMD& mat)
        {
            for (uint32_t i=0; i<VF_PLANE_COUNT; ++i)
                core::plane3dSIMDf::transform(planes[i],mat);

            recalculateBoundingBox();
        }

		//! @returns a bounding box enclosing the whole view frustum.
		inline const core::aabbox3df &getBoundingBox() const {return boundingBox;}

		//! All planes enclosing the view frustum.
		core::plane3dSIMDf planes[VF_PLANE_COUNT];

		//! Bounding box around the view frustum.
		core::aabbox3df boundingBox;
	};



	//! This constructor creates a view frustum based on a projection
	//! and/or view matrix.
	inline void SViewFrustum::setFrom(const core::matrix4SIMD& mat)
	{
	    core::vectorSIMDf lastRow = mat.getRow(3u);
		// left clipping plane
		planes[VF_LEFT_PLANE].setPlane(lastRow+mat.getRow(0u));
		// right clipping plane
		planes[VF_RIGHT_PLANE].setPlane(lastRow-mat.getRow(0u));
		// top clipping plane
		planes[VF_TOP_PLANE].setPlane(lastRow+mat.getRow(1u));
		// bottom clipping plane
		planes[VF_BOTTOM_PLANE].setPlane(lastRow-mat.getRow(1u));

		// far clipping plane
		planes[VF_FAR_PLANE].setPlane(lastRow-mat.getRow(2u));

		// near clipping plane
		planes[VF_NEAR_PLANE].setPlane(mat.getRow(2u));

		// normalize normals
		for ( auto i=0; i != VF_PLANE_COUNT; ++i)
		{
		    auto normal(planes[i].getNormal());
            *reinterpret_cast<core::vectorSIMDf*>(planes+i) *= core::inversesqrt(core::dot(normal,normal));
		}

		// make bounding box
		recalculateBoundingBox();
	}

	inline core::vector3df_SIMD SViewFrustum::getFarLeftUp() const
	{
		core::vector3df_SIMD p;
		planes[scene::SViewFrustum::VF_FAR_PLANE].getIntersectionWithPlanes(
			planes[scene::SViewFrustum::VF_TOP_PLANE],
			planes[scene::SViewFrustum::VF_LEFT_PLANE], p);

		return p;
	}

	inline core::vector3df_SIMD SViewFrustum::getFarLeftDown() const
	{
		core::vector3df_SIMD p;
		planes[scene::SViewFrustum::VF_FAR_PLANE].getIntersectionWithPlanes(
			planes[scene::SViewFrustum::VF_BOTTOM_PLANE],
			planes[scene::SViewFrustum::VF_LEFT_PLANE], p);

		return p;
	}

	inline core::vector3df_SIMD SViewFrustum::getFarRightUp() const
	{
		core::vector3df_SIMD p;
		planes[scene::SViewFrustum::VF_FAR_PLANE].getIntersectionWithPlanes(
			planes[scene::SViewFrustum::VF_TOP_PLANE],
			planes[scene::SViewFrustum::VF_RIGHT_PLANE], p);

		return p;
	}

	inline core::vector3df_SIMD SViewFrustum::getFarRightDown() const
	{
		core::vector3df_SIMD p;
		planes[scene::SViewFrustum::VF_FAR_PLANE].getIntersectionWithPlanes(
			planes[scene::SViewFrustum::VF_BOTTOM_PLANE],
			planes[scene::SViewFrustum::VF_RIGHT_PLANE], p);

		return p;
	}

	inline core::vector3df_SIMD SViewFrustum::getNearLeftUp() const
	{
		core::vector3df_SIMD p;
		planes[scene::SViewFrustum::VF_NEAR_PLANE].getIntersectionWithPlanes(
			planes[scene::SViewFrustum::VF_TOP_PLANE],
			planes[scene::SViewFrustum::VF_LEFT_PLANE], p);

		return p;
	}

	inline core::vector3df_SIMD SViewFrustum::getNearLeftDown() const
	{
		core::vector3df_SIMD p;
		planes[scene::SViewFrustum::VF_NEAR_PLANE].getIntersectionWithPlanes(
			planes[scene::SViewFrustum::VF_BOTTOM_PLANE],
			planes[scene::SViewFrustum::VF_LEFT_PLANE], p);

		return p;
	}

	inline core::vector3df_SIMD SViewFrustum::getNearRightUp() const
	{
		core::vector3df_SIMD p;
		planes[scene::SViewFrustum::VF_NEAR_PLANE].getIntersectionWithPlanes(
			planes[scene::SViewFrustum::VF_TOP_PLANE],
			planes[scene::SViewFrustum::VF_RIGHT_PLANE], p);

		return p;
	}

	inline core::vector3df_SIMD SViewFrustum::getNearRightDown() const
	{
		core::vector3df_SIMD p;
		planes[scene::SViewFrustum::VF_NEAR_PLANE].getIntersectionWithPlanes(
			planes[scene::SViewFrustum::VF_BOTTOM_PLANE],
			planes[scene::SViewFrustum::VF_RIGHT_PLANE], p);

		return p;
	}

} // end namespace scene
} // end namespace nbl

#endif

