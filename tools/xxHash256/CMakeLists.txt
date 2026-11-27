set(EXECUTABLE_NAME xxHash256)

project(${EXECUTABLE_NAME})	
add_executable(${EXECUTABLE_NAME} main.cpp)

find_package(argparse CONFIG REQUIRED)
find_package(blake3 CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED)
find_package(gtl CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Imath CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE argparse::argparse BLAKE3::blake3 boost::boost glm::glm gtl::gtl Imath::Imath nlohmann_json::nlohmann_json)
target_include_directories(${EXECUTABLE_NAME} PUBLIC 
	$<TARGET_PROPERTY:Nabla,INTERFACE_INCLUDE_DIRECTORIES> # only for xxHash256 include directory, we DO NOT want to link it nor use it
)
target_compile_definitions(${EXECUTABLE_NAME} PUBLIC "$<TARGET_PROPERTY:Nabla,INTERFACE_COMPILE_DEFINITIONS>") # workaround because must use Nabla headers without linking Nabla to itself..
				
nbl_adjust_flags(MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
nbl_adjust_definitions()