
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(BULLET_COLLISION_SRC "" CACHE STRING "For example 17.SimpleBulletIntegration - Not required")
set(BULLET_DYNAMICS_SRC "" CACHE STRING "For example 17.SimpleBulletIntegration - Not required")

set(Bullet_Extension "../../ext/Bullet3")

file(GLOB_RECURSE addFiles "${Bullet_Extension}/*" 
	"${BULLET_COLLISION_SRC}/*.cpp" "${BULLET_DYNAMICS_SRC}/*.cpp")

irr_create_executable_project("${addFiles}" "")
