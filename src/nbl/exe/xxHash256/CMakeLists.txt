set(EXECUTABLE_NAME xxHash256)

project(${EXECUTABLE_NAME})	
add_executable(${EXECUTABLE_NAME} main.cpp)

# Yas TODO: "nbl_define_h" to be removed and config generated at generate time or configure
add_dependencies(${EXECUTABLE_NAME} argparse nbl_define_h)
target_include_directories(${EXECUTABLE_NAME} PUBLIC 
	$<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>
	$<TARGET_PROPERTY:Nabla,INTERFACE_INCLUDE_DIRECTORIES> # only for xxHash256 include directory, we DO NOT want to link it nor use it
)
				
# map all to Release
nbl_adjust_flags(MAP_RELEASE Release MAP_RELWITHDEBINFO Release MAP_DEBUG Release)
nbl_adjust_definitions()