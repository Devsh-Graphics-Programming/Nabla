#ifndef _IRR_EXT_BULLET_UTILITY_INCLUDED_
#define _IRR_EXT_BULLET_UTILITY_INCLUDED_

#include "irrlicht.h"
#include <cstdint>
#include <type_traits>
#include <typeinfo>

#include "btBulletDynamicsCommon.h"

namespace irr
{
namespace ext
{
namespace Bullet3
{


   
 
   
    template<class to, class from>
    to convert(from vec) {
        static_assert(std::is_reference<to>::value && std::is_reference<from>::value, "Pass-By-Reference Assumptions Broken");
        static_assert(sizeof(to) == 16u && alignof(to) == 16u && sizeof(from) == 16u && alignof(from) == 16u,
            "Size/Alignment Assumptions When Converting Broken!");
        return reinterpret_cast<to>(vec);
    }


    inline core::vectorSIMDf &frombtVec3(btVector3 &vec) {
        return convert<core::vectorSIMDf&, btVector3&>(vec);
    }
    
    inline const core::vectorSIMDf &frombtVec3(const btVector3 &vec) {
        return convert<const core::vectorSIMDf&, const btVector3&>(vec);
    }
   
    inline core::vectorSIMDf &frombtVec4(btVector4 &vec) {
        return convert<core::vectorSIMDf&, btVector4&>(vec);
    }

    inline const core::vectorSIMDf &frombtVec4(const btVector4 &vec) {
        return convert<const core::vectorSIMDf&, const btVector4&>(vec);
    }
        
    inline btVector3 &tobtVec3(core::vectorSIMDf &vec) {
        return convert<btVector3&, core::vectorSIMDf&>(vec);
    }

    inline const btVector3 &tobtVec3(const core::vectorSIMDf &vec) {
        return convert<const btVector3&, const core::vectorSIMDf&>(vec);
    }

    inline btVector4 &tobtVec4(core::vectorSIMDf &vec) {
        return convert<btVector4&, core::vectorSIMDf&>(vec);
    }

    inline const btVector4 &tobtVec4(const core::vectorSIMDf &vec) {
        return convert<const btVector4&, const core::vectorSIMDf&>(vec);
    }

    inline core::matrix3x4SIMD convertbtTransform(const btTransform &trans) {
        core::matrix3x4SIMD mat;

        for (uint32_t i = 0; i < 3u; ++i) {
            mat.rows[i] = frombtVec3(trans.getBasis().getRow(i));
        }
        mat.setTranslation(frombtVec3(trans.getOrigin()));

        return mat;
    }

    inline btTransform convertMatrixSIMD(const core::matrix3x4SIMD &mat) {
        btTransform transform;
        
        //Calling makeSafe3D on rows erases translation so save it
        mat.getTranslation().makeSafe3D();
        btVector3 translation = tobtVec3(mat.getTranslation());



        btMatrix3x3 data;
        for (uint32_t i = 0; i < 3u; ++i) {
            //TODO - makeSafe3D()
            data[i] = tobtVec3(mat.rows[i]);
        }
        transform.setBasis(data);
        transform.setOrigin(translation);
        
        return transform;
    }


    /*
        NOTICE: REMOVE WHEN MESHSCENENODEINSTANCE IMPLEMENTS core::matrix3x4SIMD OVER core::matrix4x3
    */
    inline btTransform convertRetardedMatrix(const core::matrix4x3 &mat) {
        core::matrix3x4SIMD irrMat;
        irrMat.set(mat);
        return convertMatrixSIMD(irrMat);
    }

}
}
}



#endif