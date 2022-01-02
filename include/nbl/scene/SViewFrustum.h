#ifndef __S_VIEW_FRUSTUM_H_INCLUDED__
#define __S_VIEW_FRUSTUM_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "line3d.h"
#include "aabbox3d.h"
#include "matrix4SIMD.h"

namespace nbl::scene
{
	//! Defines the view frustum. That's the space visible by the camera.
	/** 
		The view frustum is enclosed by 6 planes. These six planes share
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

			//! Amount of planes enclosing the view frustum.
			VF_PLANE_COUNT
		};

		SViewFrustum() {}

		inline SViewFrustum(const SViewFrustum& other)
		{
			cameraPosition = other.cameraPosition;
			boundingBox = other.boundingBox;

			uint32_t i;
			for (i = 0; i < VF_PLANE_COUNT; ++i)
				planes[i] = other.planes[i];
		}

		inline SViewFrustum(const core::matrix4SIMD& mat)
		{
			setFrom(mat);
		}

		inline void setFrom(const core::matrix4SIMD& mat)
		{
			// not sure if I could use 
			// inline void setPlane(const vector3df_SIMD& point, const vector3df_SIMD& nvector)

			const auto* data = mat.pointer();
			auto test = mat[15] + mat[12];
			// left clipping plane
			{
				core::vector3df_SIMD normal(data[3] + data[0], data[7] + data[4], data[11] + data[8]);
				planes[VF_LEFT_PLANE].setNormal(normal);
				planes[VF_LEFT_PLANE].D = mat[15] + mat[12]; // ? TODO
			}

			// right clipping plane
			{
				core::vector3df_SIMD normal(data[3] - data[0], data[7] - data[4], data[11] - data[8]);
				planes[VF_RIGHT_PLANE].setNormal(normal);
				planes[VF_RIGHT_PLANE].D = mat[15] - mat[12]; // ? TODO
			}
			
			// top clipping plane
			{
				core::vector3df_SIMD normal(data[3] - data[1], data[7] - data[5], data[11] - data[9]);
				planes[VF_TOP_PLANE].setNormal(normal);
				planes[VF_TOP_PLANE].D = mat[15] - mat[13];
			}
		
			// bottom clipping plane
			{
				core::vector3df_SIMD normal(data[3] - data[1], data[7] - data[5], data[11] - data[9]);
				planes[VF_BOTTOM_PLANE].setNormal(normal);
				planes[VF_BOTTOM_PLANE].D = mat[15] + mat[13];
			}
			
			// far clipping plane
			{
				core::vector3df_SIMD normal(data[3] - data[2], data[7] - data[6], data[11] - data[10]);
				planes[VF_FAR_PLANE].setNormal(normal);
				planes[VF_FAR_PLANE].D = mat[15] - mat[14];
			}
			
			// near clipping plane
			{
				core::vector3df_SIMD normal(data[2], data[6], data[10]);
				planes[VF_NEAR_PLANE].setNormal(normal);
				planes[VF_NEAR_PLANE].D = mat[14];
			}

			// TODO: not sure if needed at all

			// normalize normals
			/*uint32_t i;
			for (i = 0; i != VF_PLANE_COUNT; ++i)
			{
				const float len = -core::reciprocal_squareroot(
					planes[i].Normal.getLengthSQ());
				planes[i].Normal *= len;
				planes[i].D *= len;
			}*/

			// make bounding box
			recalculateBoundingBox();
		}

		inline void transform(const core::matrix4SIMD& mat)
		{
			for (uint32_t i = 0; i < VF_PLANE_COUNT; ++i)
				mat.transformPlane(planes[i]); // TODO

			mat.transformVect(cameraPosition.getAsVector3df()); // TODO
			recalculateBoundingBox();
		}

		//! @returns the point which is on the far left upper corner inside the the view frustum.
		inline core::vector3df_SIMD getFarLeftUp() const
		{
			core::vector3df_SIMD p;
			planes[scene::SViewFrustum::VF_FAR_PLANE].getIntersectionWithPlanes(
				planes[scene::SViewFrustum::VF_TOP_PLANE],
				planes[scene::SViewFrustum::VF_LEFT_PLANE], p);

			return p;
		}

		//! @returns the point which is on the far left bottom corner inside the the view frustum.
		inline core::vector3df_SIMD getFarLeftDown() const
		{
			core::vector3df_SIMD p;
			planes[scene::SViewFrustum::VF_FAR_PLANE].getIntersectionWithPlanes(
				planes[scene::SViewFrustum::VF_BOTTOM_PLANE],
				planes[scene::SViewFrustum::VF_LEFT_PLANE], p);

			return p;
		}

		//! @returns the point which is on the far right top corner inside the the view frustum.
		inline core::vector3df_SIMD getFarRightUp() const
		{
			core::vector3df_SIMD p;
			planes[scene::SViewFrustum::VF_FAR_PLANE].getIntersectionWithPlanes(
				planes[scene::SViewFrustum::VF_TOP_PLANE],
				planes[scene::SViewFrustum::VF_RIGHT_PLANE], p);

			return p;
		}

		//! @returns the point which is on the far right bottom corner inside the the view frustum.
		inline core::vector3df_SIMD getFarRightDown() const
		{
			core::vector3df_SIMD p;
			planes[scene::SViewFrustum::VF_FAR_PLANE].getIntersectionWithPlanes(
				planes[scene::SViewFrustum::VF_BOTTOM_PLANE],
				planes[scene::SViewFrustum::VF_RIGHT_PLANE], p);

			return p;
		}

		//! @returns the point which is on the near left upper corner inside the the view frustum.
		inline core::vector3df_SIMD getNearLeftUp() const
		{
			core::vector3df_SIMD p;
			planes[scene::SViewFrustum::VF_NEAR_PLANE].getIntersectionWithPlanes(
				planes[scene::SViewFrustum::VF_TOP_PLANE],
				planes[scene::SViewFrustum::VF_LEFT_PLANE], p);

			return p;
		}

		//! @returns the point which is on the near left bottom corner inside the the view frustum.
		inline core::vector3df_SIMD getNearLeftDown() const
		{
			core::vector3df_SIMD p;
			planes[scene::SViewFrustum::VF_NEAR_PLANE].getIntersectionWithPlanes(
				planes[scene::SViewFrustum::VF_BOTTOM_PLANE],
				planes[scene::SViewFrustum::VF_LEFT_PLANE], p);

			return p;
		}

		//! @returns the point which is on the near right top corner inside the the view frustum.
		inline core::vector3df_SIMD getNearRightUp() const
		{
			core::vector3df_SIMD p;
			planes[scene::SViewFrustum::VF_NEAR_PLANE].getIntersectionWithPlanes(
				planes[scene::SViewFrustum::VF_TOP_PLANE],
				planes[scene::SViewFrustum::VF_RIGHT_PLANE], p);

			return p;
		}

		//! @returns the point which is on the near right bottom corner inside the the view frustum.
		inline core::vector3df_SIMD getNearRightDown() const
		{
			core::vector3df_SIMD p;
			planes[scene::SViewFrustum::VF_NEAR_PLANE].getIntersectionWithPlanes(
				planes[scene::SViewFrustum::VF_BOTTOM_PLANE],
				planes[scene::SViewFrustum::VF_RIGHT_PLANE], p);

			return p;
		}

		//! @returns a bounding box enclosing the whole view frustum.
		inline const core::aabbox3d<float>& getBoundingBox() const
		{
			return boundingBox;
		}

		//! Recalculates the bounding box member based on the planes.
		inline void recalculateBoundingBox()
		{
			boundingBox.reset(cameraPosition.getAsVector3df());

			boundingBox.addInternalPoint(getFarLeftUp().getAsVector3df());
			boundingBox.addInternalPoint(getFarRightUp().getAsVector3df());
			boundingBox.addInternalPoint(getFarLeftDown().getAsVector3df());
			boundingBox.addInternalPoint(getFarRightDown().getAsVector3df());
		}

		//! Clips a line to the view frustum.
		/** @return True if the line was clipped, false if not. */
		inline bool clipLine(core::line3d<float>& line) const
		{
			bool wasClipped = false;
			for (uint32_t i = 0; i < VF_PLANE_COUNT; ++i)
			{
				if (planes[i].classifyPointRelation(line.start) == core::ISREL3D_FRONT)
				{
					line.start.interpolate(line.start, line.end, planes[i].getKnownIntersectionWithLine(line.start, line.end));
					wasClipped = true;
				}
				if (planes[i].classifyPointRelation(line.end) == core::ISREL3D_FRONT)
				{
					line.end.interpolate(line.start, line.end, planes[i].getKnownIntersectionWithLine(line.start, line.end));
					wasClipped = true;
				}
			}
			return wasClipped;
		}

		//! Is point lying within the frustum?
		/** @returns Whether point is lying within the frustum. */
		inline bool cullPoint(const core::vector3d<float>& point) const
		{
			for (uint32_t i = 0; i < VF_PLANE_COUNT; ++i)
			{
				if (planes[i].classifyPointRelation(point) == core::ISREL3D_FRONT)
					return true;
			}
			return false;
		}

		//! The position of the camera.
		core::vector3df_SIMD cameraPosition;

		//! All planes enclosing the view frustum.
		core::plane3dSIMDf planes[VF_PLANE_COUNT];

		//! Bounding box around the view frustum.
		core::aabbox3d<float> boundingBox;
	};
} // nbl::scene

#endif

