set(BOOST_PREPROCESSOR_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/superproject/libs/preprocessor/include" CACHE PATH "" FORCE)

get_filename_component(_BOOST_PREPROCESSOR_BR_BUNDLE_SEARCH_DIRECTORY_ "${BOOST_PREPROCESSOR_INCLUDE}" ABSOLUTE)
get_filename_component(_BOOST_PREPROCESSOR_BR_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
get_filename_component(_BOOST_PREPROCESSOR_BR_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)
set(BOOST_BUILTIN_RESOURCES_DIRECTORY_PATH "${_BOOST_PREPROCESSOR_BR_BUNDLE_SEARCH_DIRECTORY_}" CACHE INTERNAL "" FORCE)

if(NBL_EMBED_BUILTIN_RESOURCES)
	include("${NBL_ROOT_PATH}/src/nbl/builtin/utils.cmake")
	
	file(GLOB_RECURSE BOOST_HEADERS_REC_REL RELATIVE "${BOOST_PREPROCESSOR_INCLUDE}/boost" "${BOOST_PREPROCESSOR_INCLUDE}/*")
	
	foreach(BOOST_HEADER_REL IN LISTS BOOST_HEADERS_REC_REL)
		LIST_BUILTIN_RESOURCE(BOOST_RESOURCES_TO_EMBED "${BOOST_HEADER_REL}")
	endforeach()

    ADD_CUSTOM_BUILTIN_RESOURCES(boostBuiltinResourceData BOOST_RESOURCES_TO_EMBED "${_BOOST_PREPROCESSOR_BR_BUNDLE_SEARCH_DIRECTORY_}" "boost" "boost::builtin" "${_BOOST_PREPROCESSOR_BR_OUTPUT_DIRECTORY_HEADER_}" "${_BOOST_PREPROCESSOR_BR_OUTPUT_DIRECTORY_HEADER_}" "STATIC" "INTERNAL")
endif()

set(BOOST_INCLUDE_LIBRARIES wave)
set(BOOST_STAGEDIR "${CMAKE_CURRENT_BINARY_DIR}/boost/superproject/stage")
add_subdirectory(superproject superproject EXCLUDE_FROM_ALL)

set(NBL_BOOST_LIBS wave;assert;concept_check;config;core;filesystem;format;iterator;lexical_cast;mpl;multi_index;optional;pool;preprocessor;serialization;smart_ptr;spirit;static_assert;throw_exception;type_traits;atomic;container_hash;detail;io;predef;system;winapi;utility;conversion;function_types;fusion;container;integer;numeric/conversion;bind;move;tuple;array;function;mp11;variant;variant2;endian;phoenix;proto;range;regex;thread;typeof;unordered;align;intrusive;describe;functional;chrono;date_time;exception;type_index;ratio;algorithm;tokenizer)

foreach(BOOST_LIB IN LISTS NBL_BOOST_LIBS)
if(TARGET boost_${BOOST_LIB})
	list(APPEND NBL_BOOST_TARGETS boost_${BOOST_LIB})
endif()
endforeach()

set(NBL_BOOST_TARGETS 
	${NBL_BOOST_TARGETS}
PARENT_SCOPE)
