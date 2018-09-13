#ifndef __S_ELLIPSOID_COLLIDER_H_INCLUDED__
#define __S_ELLIPSOID_COLLIDER_H_INCLUDED__

#include "vectorSIMD.h"

namespace irr
{
namespace core
{

class SEllipsoidCollider : public AllocationOverrideDefault
{
        vectorSIMDf negativeCenter;
        vectorSIMDf reciprocalAxes;
    public:
        SEllipsoidCollider(bool& validEllipse, const vectorSIMDf& centr, const vectorSIMDf& axisLengths) : negativeCenter(-centr)
        {
            for (size_t i=0; i<3; i++)
            {
                if (axisLengths.pointer[i]==0.f)
                {
                    validEllipse = false;
                    return;
                }
            }

            reciprocalAxes = reciprocal(axisLengths);

            validEllipse = true;
        }

        inline bool CollideWithRay(float& collisionDistance, vectorSIMDf origin, vectorSIMDf direction, const float& dirMaxMultiplier) const
        {
            origin += negativeCenter;

            vectorSIMDf originLen2 = dot(origin,origin);
            if (originLen2.X<=1.f) //point is inside
            {
                collisionDistance = 0.f;
                return true;
            }

            direction *= reciprocalAxes;
            vectorSIMDf dirLen2 = dot(direction,direction);
            vectorSIMDf dirDotOrigin = dot(direction,origin);

            vectorSIMDf determinant = dirDotOrigin+dirLen2*originLen2-dirLen2;
            if (determinant.X<0.f)
                return false;

            determinant = dirDotOrigin+sqrt(determinant);
            if (determinant.X>0.f)
                return false;

            vectorSIMDf t = -determinant*reciprocal(dirLen2);

            if (t.X<dirMaxMultiplier)
            {
                collisionDistance = t.X;
                return true;
            }
            else
                return false;
        }
};


}
}

#endif

