#include "../../ext/Bullet3/BulletUtility.h"

namespace irr
{
namespace ext 
{

namespace Bullet3 {

core::vector3df btVec3Convert(btVector3 vec) {
    return core::vector3df(vec.m_floats);
}

btVector3 irrVec3Convert(core::vector3df vec) {
    return btVector3(vec.X, vec.Y, vec.Z);
}

}
}
}