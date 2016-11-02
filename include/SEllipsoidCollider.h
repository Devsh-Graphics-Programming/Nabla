#ifndef __S_ELLIPSOID_COLLIDER_H_INCLUDED__
#define __S_ELLIPSOID_COLLIDER_H_INCLUDED__

#include "vectorSIMD.h"

namespace irr
{
namespace core
{

class SEllipsoidCollider
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


        static inline void* operator new(size_t size) throw(std::bad_alloc)
        {
            void *memoryallocatedaligned = 0;
#ifdef _IRR_WINDOWS_
            memoryallocatedaligned = _aligned_malloc(size,SIMD_ALIGNMENT);
#else
            posix_memalign((void**)&memoryallocatedaligned,SIMD_ALIGNMENT,size);
#endif
            return memoryallocatedaligned;
        }
        static inline void operator delete(void* ptr)
        {
#ifdef _IRR_WINDOWS_
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
        static inline void* operator new[](size_t size) throw(std::bad_alloc)
        {
            void *memoryallocatedaligned = 0;
#ifdef _IRR_WINDOWS_
            memoryallocatedaligned = _aligned_malloc(size,SIMD_ALIGNMENT);
#else
            posix_memalign((void**)&memoryallocatedaligned,SIMD_ALIGNMENT,size);
#endif
            return memoryallocatedaligned;
        }
        static inline void  operator delete[](void* ptr) throw()
        {
#ifdef _IRR_WINDOWS_
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
        static inline void* operator new(std::size_t size,void* p) throw(std::bad_alloc)
        {
            return p;
        }
        static inline void  operator delete(void* p,void* t) throw() {}
        static inline void* operator new[](std::size_t size,void* p) throw(std::bad_alloc)
        {
            return p;
        }
        static inline void  operator delete[](void* p,void* t) throw() {}


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

