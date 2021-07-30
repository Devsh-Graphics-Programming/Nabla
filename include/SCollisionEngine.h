// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_S_COLLISION_ENGINE_H_INCLUDED__
#define __NBL_S_COLLISION_ENGINE_H_INCLUDED__

#include "nabla.h"
#include "SCompoundCollider.h"
#include "SViewFrustum.h"

namespace nbl
{
namespace core
{

class SCollisionEngine : public AllocationOverrideDefault
{
        core::vector<core::smart_refctd_ptr<SCompoundCollider> > colliders;

    public:
		//! Destructor.
		~SCollisionEngine() = default;

#if 0
		//! Returns a 3d ray which would go through the 2d screen coodinates.
		/**
		@param[out] origin Start point point of the output ray
		@param[out] direction Normalized vector denoting direction of the output ray
		@param[out] rayLen Length of the output ray
		@param[in] uv Screen coordinates
		@param[in] driver Driver; needed to get size of viewport
		@param[in] camera Camera on which calculations will depend
		*/
		inline static bool getRayFromScreenCoordinates(vectorSIMDf &origin, vectorSIMDf &direction, float& rayLen,
                                        const position2di& uv, video::IVideoDriver* driver, scene::ICameraSceneNode* camera)
        {
            if (!camera||!driver)
                return false;

            const scene::SViewFrustum* f = camera->getViewFrustum();

            vector3df_SIMD farLeftUp = f->getFarLeftUp();
            vector3df_SIMD lefttoright = f->getFarRightUp() - farLeftUp;
            vector3df_SIMD uptodown = f->getFarLeftDown() - farLeftUp;

            const rect<int32_t>& viewPort = driver->getViewPort();
            dimension2d<uint32_t> screenSize(viewPort.getWidth(), viewPort.getHeight());

            float dx = uv.X;
            dx /= (float)screenSize.Width;
            float dy = uv.Y;
            dy /= (float)screenSize.Height;

            if (camera->isOrthogonal())
                origin = f->cameraPosition + lefttoright * (dx-0.5f) + uptodown * (dy-0.5f);
            else
                origin = f->cameraPosition;

            direction.set(farLeftUp + lefttoright * dx + uptodown * dy);
            direction -= origin;
            rayLen = length(direction).X;
            direction /= rayLen;
            return true;
        }

		//! Calculates 2d screen position from a 3d position.
		/**
		@param pos 3d position which is to be projected on screen
		@param driver Driver
		@param camera Camera on which calculations will depend
		@param iseViewPort Whether to use viewport or current render target's size
		@returns 2d position or {-100000, -100000} (minus ten thousand) if the point is behind camera.
		*/
		inline static position2di getScreenCoordinatesFrom3DPosition(const vector3df& pos, video::IVideoDriver* driver, scene::ICameraSceneNode* camera, bool useViewPort=false)
		{
            if (!driver||!camera)
                return position2d<int32_t>(-100000,-100000);

            dimension2d<uint32_t> dim;
            if (useViewPort)
                dim.set(driver->getViewPort().getWidth(), driver->getViewPort().getHeight());
            else
                dim=(driver->getCurrentRenderTargetSize());

            dim.Width /= 2;
            dim.Height /= 2;

            auto trans = camera->getConcatenatedMatrix();

            core::vectorSIMDf transformedPos(pos.X, pos.Y, pos.Z, 1.0f );

            trans.transformVect(transformedPos);

            if (transformedPos.w < 0)
                return position2d<int32_t>(-10000,-10000);

            const float zDiv = transformedPos.w==0.f  ?  1.f:reciprocal_approxim(transformedPos).w;

            return position2d<int32_t>(
                        dim.Width + round<float,int32_t>(dim.Width * (transformedPos.x * zDiv)),
                        dim.Height - round<float,int32_t>(dim.Height * (transformedPos.y * zDiv)));
		}
#endif // 0

		//! Adds a collider
		/** @param collider A pointer to collider. */
        inline void addCompoundCollider(core::smart_refctd_ptr<SCompoundCollider>&& collider)
        {
            if (!collider)
                return;

            auto found = std::lower_bound(colliders.begin(),colliders.end(),collider);
            if (found!=colliders.end() && *found==collider)
                return;

            colliders.insert(found,std::move(collider));
        }

		//! Removes collider pointed by `collider`
		/** @param collider Pointer to collider. s*/
        inline void removeCompoundCollider(const SCompoundCollider* collider)
        {
            if (!collider)
                return;

            auto found = std::lower_bound(colliders.begin(),colliders.end(),reinterpret_cast<const core::smart_refctd_ptr<SCompoundCollider>&>(collider));
            if (found==colliders.end() || found->get()!=collider)
			{
//				FW_WriteToLog(kLogError,"removeCompoundCollider collider not found!\n");
                return;
			}

            colliders.erase(found);
        }

		//! Gets current amount of colliders
		/** @rturns Current amount of colliders. */
        inline size_t getColliderCount() const { return colliders.size(); }

		//! Performs collision test with a given ray defined by `origin`, `direction` and `maxRayLen` parameters
		/**
		@param[out] hitPointObjectData Data of collider with which the collision occured. Does not get touched if no collision occured.
		@param[out] collisionDistance If no collision occured - gets value of `maxRayLen` parameter. Otherwise - ???
		@param[in] origin Start point point of the input ray
		@param[in] direction Normalized vector denoting direction of the input ray
		@param[in] maxRayLen Length of the input ray
		*/
        inline bool FastCollide(SColliderData& hitPointObjectData, float &collisionDistance, const vectorSIMDf& origin, const vectorSIMDf& direction, const float& maxRayLen=FLT_MAX) const
        {
            bool retval = false;

            collisionDistance = maxRayLen;
            for (size_t i=0; i<colliders.size(); i++)
            {
                float tmpDist;
                if (colliders[i]->CollideWithRay(tmpDist,origin,direction,collisionDistance)&&tmpDist<collisionDistance)
                {
                    collisionDistance = tmpDist;
                    hitPointObjectData = colliders[i]->getColliderData();
                    retval = true;
                }
            }

            return retval;
        }
};

}
}

#endif
