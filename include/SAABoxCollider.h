#ifndef __S_AXIS_ALIGNED_BOX_COLLIDER_H_INCLUDED__
#define __S_AXIS_ALIGNED_BOX_COLLIDER_H_INCLUDED__

#include "vectorSIMD.h"
#include "aabbox3d.h"

namespace irr
{
namespace core
{

class SAABoxCollider : public AllocationOverrideDefault
{
    public:
        SAABoxCollider(const aabbox3df& box) : Box(box) {}


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
            vectorSIMDf t = (MinEdgeSSE&_mm_castsiWRONGCASTTALKTODEVSH128_ps(xmm0.getAsRegister()))|(MaxEdgeSSE&_mm_castsiWRONGCASTTALKTODEVSH128_ps(xmm1.getAsRegister())); // SSE AND,AND,OR
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
            vectorSIMDf t_Min = MinEdgeSSE&xmm0.getAsRegister();
            vectorSIMDf t_Max = MaxEdgeSSE&xmm0.getAsRegister();
            xmm0 = ~xmm0;
            t_Min = t_Min|(reinterpret_cast<const vectorSIMDu32&>(MaxEdgeSSE)&xmm0.getAsRegister());
            t_Max = t_Max|(reinterpret_cast<const vectorSIMDu32&>(MinEdgeSSE)&xmm0.getAsRegister());

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
