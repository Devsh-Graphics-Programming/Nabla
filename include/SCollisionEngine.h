#ifndef __S_COLLISION_ENGINE_H_INCLUDED__
#define __S_COLLISION_ENGINE_H_INCLUDED__

#include "irrlicht.h"
#include "SCompoundCollider.h"
#include "SViewFrustum.h"

namespace irr
{
namespace core
{

class SCollisionEngine
{
        array<SCompoundCollider*> colliders;

    public:
        ~SCollisionEngine()
        {
			for (size_t i=0; i<colliders.size(); i++)
				colliders[i]->drop();
        }

		//! Returns a 3d ray which would go through the 2d screen coodinates.
		inline static bool getRayFromScreenCoordinates(vectorSIMDf &origin, vectorSIMDf &direction, float& rayLen,
                                        const position2di& uv, video::IVideoDriver* driver, scene::ICameraSceneNode* camera)
        {
            if (!camera||!driver)
                return false;

            const scene::SViewFrustum* f = camera->getViewFrustum();

            vector3df farLeftUp = f->getFarLeftUp();
            vector3df lefttoright = f->getFarRightUp() - farLeftUp;
            vector3df uptodown = f->getFarLeftDown() - farLeftUp;

            const rect<int32_t>& viewPort = driver->getViewPort();
            dimension2d<uint32_t> screenSize(viewPort.getWidth(), viewPort.getHeight());

            float dx = uv.X;
            dx /= (float)screenSize.Width;
            float dy = uv.Y;
            dy /= (float)screenSize.Height;

            if (camera->isOrthogonal())
                origin.set(f->cameraPosition + lefttoright * (dx-0.5f) + uptodown * (dy-0.5f));
            else
                origin.set(f->cameraPosition);

            direction.set(farLeftUp + lefttoright * dx + uptodown * dy);
            direction -= origin;
            rayLen = length(direction).X;
            direction /= rayLen;
            return true;
        }

		//! Calculates 2d screen position from a 3d position.
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

            matrix4 trans = camera->getConcatenatedMatrix();

            float transformedPos[4] = { pos.X, pos.Y, pos.Z, 1.0f };

            trans.multiplyWith1x4Matrix(transformedPos);

            if (transformedPos[3] < 0)
                return position2d<int32_t>(-10000,-10000);

            const float zDiv = transformedPos[3]==0.f  ?  1.f:reciprocal(transformedPos[3]);

            return position2d<int32_t>(
                        dim.Width + round32(dim.Width * (transformedPos[0] * zDiv)),
                        dim.Height - round32(dim.Height * (transformedPos[1] * zDiv)));
		}

        inline void addCompoundCollider(SCompoundCollider* collider)
        {
            if (!collider)
                return;

            collider->grab();
            colliders.push_back(collider);
        }

        inline void removeCompoundCollider(SCompoundCollider* collider)
        {
            if (!collider)
                return;

            int32_t ix = colliders.binary_search(collider);
            if (ix<0)
			{
//				FW_WriteToLog(kLogError,"removeCompoundCollider collider not found!\n");
                return;
			}

			collider->drop();
            colliders.erase(ix);
        }

        inline size_t getColliderCount() const { return colliders.size(); }

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
