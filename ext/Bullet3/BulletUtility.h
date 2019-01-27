#ifndef _IRR_EXT_BULLET_UTILITY_INCLUDED_
#define _IRR_EXT_BULLET_UTILITY_INCLUDED_

#include "irrlicht.h"
#include <cstdint>
#include <type_traits>

#include "btBulletDynamicsCommon.h"

namespace irr
{
namespace ext
{
namespace Bullet3
{

    core::matrix3x4SIMD &convertbtTransform(btTransform &transform);


    template<class to>
    to &convertFromVecSIMDf(core::vectorSIMDf &vec) {
        static_assert(sizeof(to) == 16u && alignof(to) == 16u, "from vectorSIMDf conversion - Size and Alignment Assumptions violated!");
        return reinterpret_cast<to&>(vec);
    }
    template<class to>
    core::vectorSIMDf &convertToVecSIMDf(to &vec) {
        static_assert(sizeof(to) == 16u && alignof(to) == 16u, "to vectorSIMDf conversion - Size and Alignment Assumptions violated!");
        return reinterpret_cast<core::vectorSIMDf&>(vec);
    }

    inline core::vectorSIMDf &btVec3Convert(btVector3 &vec) {
        return convertToVecSIMDf<btVector3>(vec);
    }
    
    inline core::vectorSIMDf &btVec4Convert(btVector4 &vec) {
        return convertToVecSIMDf<btVector4>(vec);
    }

    inline btVector3 &SIMDfConvertToVec3(core::vectorSIMDf &vec) {
        return convertFromVecSIMDf<btVector3&>(vec);
    }

    inline btVector4 &SIMDfConvertToVec4(core::vectorSIMDf &vec) {
        return convertFromVecSIMDf<btVector4&>(vec);
    }


}
}
}



#endif