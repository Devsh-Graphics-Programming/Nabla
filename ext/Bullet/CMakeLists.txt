file(GLOB_RECURSE BULLETCOLLISION_CPP ${BULLET_INCLUDE_PATH}/BulletCollision/*.cpp)
file(GLOB_RECURSE BULLETDYNAMICS_CPP ${BULLET_INCLUDE_PATH}/BulletDynamics/*.cpp)
file(GLOB_RECURSE LINEARMATH_CPP ${BULLET_INCLUDE_PATH}/LinearMath/*.cpp)


file(GLOB_RECURSE BULLETCOLLISION_H ${BULLET_INCLUDE_PATH}/BulletCollision/*.h)
file(GLOB_RECURSE BULLETDYNAMICS_H ${BULLET_INCLUDE_PATH}/BulletDynamics/*.h)
file(GLOB_RECURSE LINEARMATH_H ${BULLET_INCLUDE_PATH}/LinearMath/*.h)



set(IRR_EXT_BULLET_H
  ${BULLETCOLLISION_H}
  ${BULLETDYNAMICS_H}
  ${LINEARMATH_H}
  BulletUtility.h 
  CDebugRender.h 
  CInstancedMotionState.h
  CPhysicsWorld.h
  IMotionStateBase.h
)

set(IRR_EXT_BULLET_SRC
  ${BULLETCOLLISION_CPP}
  ${BULLETDYNAMICS_CPP}
  ${LINEARMATH_CPP}
  BulletUtility.cpp
  CDebugRender.cpp
  CInstancedMotionState.cpp
  CPhysicsWorld.cpp
)

set(IRR_EXT_BULLET_EXTERNAL_INCLUDE
  "${BULLET_INCLUDE_PATH}"
)

irr_create_ext_library_project(
  BULLET
  "${IRR_EXT_BULLET_H}"
  "${IRR_EXT_BULLET_SRC}"
  "${IRR_EXT_BULLET_EXTERNAL_INCLUDE}"
  ""
)

