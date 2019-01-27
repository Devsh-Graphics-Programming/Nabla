#include "../../ext/Bullet3/BulletUtility.h"

#include <iostream>

namespace irr
{
namespace ext 
{

namespace Bullet3 
{

    /*
        01 02 03 00 | 04 05 06 00 | 07 08 09 00 | 10 11 12 00
        
        05 02 08 10 | 04 01 07 11 | 06 03 09 12 
    */

    core::matrix3x4SIMD &convertbtTransform(btTransform &transform) {
        core::matrix3x4SIMD mat;
        float *data = (float*)(&transform);
      

        btQuaternion quat = transform.getRotation();
        
       // mat.setRotation(irr::core::quaternion((float*)(&quat)));
      
        return mat;
    }

}
}
}