
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()


set(BULLET_SRC "" CACHE STRING "For example 17.SimpleBulletIntegration - Not required")
set(BULLET_INCLUDE_DIR "" CACHE STRING "For example 17.SimpleBulletIntegration - Not required")

set(Bullet_Extension "../../ext/Bullet3")
set(DebugDraw_Extension "../../ext/Bullet3")

file(GLOB_RECURSE addFiles "${Bullet_Extension}/*"  "${DebugDraw_Extension}/*.cpp"
	"${BULLET_SRC}/*.cpp")

irr_create_executable_project("${addFiles}" "")
target_include_directories(${EXECUTABLE_NAME} PUBLIC ${BULLET_INCLUDE_DIR})