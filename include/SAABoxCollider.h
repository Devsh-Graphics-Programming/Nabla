#ifndef __S_AXIS_ALIGNED_BOX_COLLIDER_H_INCLUDED__
#define __S_AXIS_ALIGNED_BOX_COLLIDER_H_INCLUDED__

#include "vectorSIMD.h"
#include "aabbox3d.h"

namespace irr
{
namespace core
{

class SAABoxCollider
{
    public:
        SAABoxCollider(const aabbox3df& box) : Box(box) {}


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



        inline bool CollideWithRay(float& collisionDistance, const vectorSIMDf& origin, const vectorSIMDf& direction, const float& dirMaxMultiplier, const vectorSIMDf& reciprocalDirection) const
        {
            if (Box.isPointInside(origin.getAsVector3df()))
            {
                collisionDistance = 0.f;
                return true;
            }

            vectorSIMDf MinEdgeSSE;
            MinEdgeSSE.set(Box.MinEdge);
            vectorSIMDf MaxEdgeSSE;
            MaxEdgeSSE.set(Box.MaxEdge);
/*
            vectorSIMDBool<4> xmm0 = reciprocalDirection>vectorSIMDf(0.f);
            vectorSIMDBool<4> xmm1 = ~xmm0; // SSE GRTER, SSE XOR
            vectorSIMDf t = (MinEdgeSSE&_mm_castsi128_ps(xmm0.getAsRegister()))|(MaxEdgeSSE&_mm_castsi128_ps(xmm1.getAsRegister())); // SSE AND,AND,OR
            t -= origin;
            t *= reciprocalDirection;

            float finalT = -FLT_MAX; //just under -0.f
            for (size_t i=0; i<3; i++)
            {
                if (t.pointer[i]>finalT&&t.pointer[i]<=dirMaxMultiplier)
                    finalT = t.pointer[i];
            }

            if (finalT<0.f)
                return false;

            vectorSIMDf outPoint = direction*(finalT+0.00000001f)+origin; //no offsets, BAAADD
            if (Box.isPointInside(outPoint.getAsVector3df()))
            {
                collisionDistance = finalT;
                return true;
            }
            else
                return false;
*/


            MinEdgeSSE -= origin;
            MaxEdgeSSE -= origin; // 2x SSE SUB
            MinEdgeSSE *= reciprocalDirection;
            MaxEdgeSSE *= reciprocalDirection; //2x SSE MUL

            vectorSIMDBool<4> xmm0 = reciprocalDirection>vectorSIMDf(0.f);
            vectorSIMDf t_Min = MinEdgeSSE&_mm_castsi128_ps(xmm0.getAsRegister());
            vectorSIMDf t_Max = MaxEdgeSSE&_mm_castsi128_ps(xmm0.getAsRegister());
            xmm0 = ~xmm0;
            t_Min = t_Min|(MaxEdgeSSE&_mm_castsi128_ps(xmm0.getAsRegister()));
            t_Max = t_Max|(MinEdgeSSE&_mm_castsi128_ps(xmm0.getAsRegister()));

            if (t_Min.pointer[1]>t_Min.pointer[0])
                t_Min.pointer[0] = t_Min.pointer[1];
            if (t_Min.pointer[2]>t_Min.pointer[0])
                t_Min.pointer[0] = t_Min.pointer[2];

            if (t_Min.pointer[0]>=dirMaxMultiplier||t_Min.pointer[0]<0.f)
                return false;

            if (t_Max.pointer[1]<t_Max.pointer[0])
                t_Max.pointer[0] = t_Max.pointer[1];
            if (t_Max.pointer[2]<t_Max.pointer[0])
            {
                if (t_Min.pointer[0]<=t_Max.pointer[2])
                {
                    collisionDistance = t_Min.pointer[0];
                    return true;
                }
                return false;
            }
            else if (t_Min.pointer[0]<=t_Max.pointer[0])
            {
                collisionDistance = t_Min.pointer[0];
                return true;
            }
            return false;
        }


        aabbox3df Box;
};


}
}

#endif
