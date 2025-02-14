// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_BULLET_UTILITY_INCLUDED_
#define _NBL_EXT_BULLET_UTILITY_INCLUDED_

#include "nabla.h"
#include <cstdint>
#include <type_traits>
#include <typeinfo>

#include "btBulletDynamicsCommon.h"

namespace nbl
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


    inline hlsl::float32_t4 &frombtVec3(btVector3 &vec) {
        return convert<hlsl::float32_t4&, btVector3&>(vec);
    }
    
    inline const hlsl::float32_t4 &frombtVec3(const btVector3 &vec) {
        return convert<const hlsl::float32_t4&, const btVector3&>(vec);
    }
   
    inline hlsl::float32_t4 &frombtVec4(btVector4 &vec) {
        return convert<hlsl::float32_t4&, btVector4&>(vec);
    }

    inline const hlsl::float32_t4 &frombtVec4(const btVector4 &vec) {
        return convert<const hlsl::float32_t4&, const btVector4&>(vec);
    }
        
    inline btVector3 &tobtVec3(hlsl::float32_t4 &vec) {
        return convert<btVector3&, hlsl::float32_t4&>(vec);
    }

    inline const btVector3 &tobtVec3(const hlsl::float32_t4 &vec) {
        return convert<const btVector3&, const hlsl::float32_t4&>(vec);
    }

    inline btVector4 &tobtVec4(hlsl::float32_t4 &vec) {
        return convert<btVector4&, hlsl::float32_t4&>(vec);
    }

    inline const btVector4 &tobtVec4(const hlsl::float32_t4 &vec) {
        return convert<const btVector4&, const hlsl::float32_t4&>(vec);
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

}
}
}



#endif