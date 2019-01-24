#ifndef _IRR_EXT_BULLET_UTILITY_INCLUDED_
#define _IRR_EXT_BULLET_UTILITY_INCLUDED_

#include "irrlicht.h"
#include <cstdint>

#include "btBulletDynamicsCommon.h"

namespace irr
{
namespace ext
{
namespace Bullet3
{
    core::vector3df btVec3Convert(btVector3 vec);
    btVector3 irrVec3Convert(core::vector3df vec);


}
}
}


#endif