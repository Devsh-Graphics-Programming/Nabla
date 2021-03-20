# Try to find OptiX project dll/so and headers
#

# outputs
unset(OPTIX_FOUND CACHE)
unset(OPTIX_INCLUDE_DIR CACHE)

# OPTIX_LOCATION can be setup to search versions somewhere else

macro ( folder_list result curdir substring )
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*${substring}*)
  SET(dirlist "")
  foreach ( child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

macro(_find_version_path targetVersion targetPath rootName searchList )
  unset ( targetVersion )
  unset ( targetPath )
  SET ( bestver "0.0.0" )
  SET ( bestpath "" )
  foreach ( basedir ${searchList} )
    folder_list ( dirList ${basedir} ${rootName} )
	  foreach ( checkdir ${dirList} ) 	 
	    string ( REGEX MATCH "${rootName}(.*)([0-9]+).([0-9]+).([0-9]+)(.*)$" result "${checkdir}" )
	    if ( "${result}" STREQUAL "${checkdir}" )
	       # found a path with versioning 
          if(${CMAKE_MATCH_2} LESS "7" OR ${CMAKE_MATCH_2} EQUAL "7" AND ${CMAKE_MATCH_3} LESS "2") 
		   message(STATUS "Found OptiX ${CMAKE_MATCH_2}.${CMAKE_MATCH_3}.${CMAKE_MATCH_4}. Expecting 7.2.0 or higher")
           continue()
          endif()
	       SET ( ver "${CMAKE_MATCH_2}.${CMAKE_MATCH_3}.${CMAKE_MATCH_4}" )
	       if ( ver GREATER bestver )
	  	    SET ( bestver ${ver} )
          SET ( bestmajorver ${CMAKE_MATCH_2})
          SET ( bestminorver ${CMAKE_MATCH_3})
	  		SET ( bestpath "${basedir}/${checkdir}" )
	  	 endif ()
	    endif()	  
	  endforeach ()		
  endforeach ()  
  SET ( ${targetVersion} "${bestver}" )
  SET ( ${targetPath} "${bestpath}" )
endmacro()

macro(_find_files targetVar incDir dllName dllName64 folder)
  unset ( fileList )
  if(ARCH STREQUAL "x86")
      file(GLOB fileList "${${incDir}}/../${folder}${dllName}")
      list(LENGTH fileList NUMLIST)
      if(NUMLIST EQUAL 0)
        file(GLOB fileList "${${incDir}}/${folder}${dllName}")
      endif()
  else()
      file(GLOB fileList "${${incDir}}/../${folder}${dllName64}")
      list(LENGTH fileList NUMLIST)
      if(NUMLIST EQUAL 0)
        file(GLOB fileList "${${incDir}}/${folder}${dllName64}")
      endif()
  endif()  
  list(LENGTH fileList NUMLIST)
  if(NUMLIST EQUAL 0)
    message(STATUS "MISSING: unable to find ${targetVar} files (${folder}${dllName}, ${folder}${dllName64})" )
    set (${targetVar} "NOTFOUND")    
  endif()
  list(APPEND ${targetVar} ${fileList} )  

  # message ( "File list: ${${targetVar}}" )		#-- debugging
endmacro()

if (DEFINED OPTIX_LOCATION OR DEFINED ENV{OPTIX_LOCATION} )
  Message(STATUS "using OPTIX_LOCATION (${OPTIX_LOCATION})...")
  if(NOT DEFINED OPTIX_LOCATION)
    if(DEFINED ENV{OPTIX_LOCATION})
      set(OPTIX_LOCATION $ENV{OPTIX_LOCATION})
    endif()
  endif()
  # Locate by version failed. Handle user override for OPTIX_LOCATION.
  string ( REGEX MATCH ".*([0-9]+).([0-9]+).([0-9]+)(.*)$" result "${OPTIX_LOCATION}" )
  if ( "${result}" STREQUAL "${OPTIX_LOCATION}" )
	if(${CMAKE_MATCH_2} LESS "7" OR ${CMAKE_MATCH_2} EQUAL "7" AND ${CMAKE_MATCH_3} LESS "2") 
      message(STATUS "Found OptiX ${CMAKE_MATCH_2}.${CMAKE_MATCH_3}.${CMAKE_MATCH_4}. Expecting 7.2.0 or higher")
      continue()
    else()
    SET ( bestver "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}" )
    SET ( bestmajorver ${CMAKE_MATCH_1})
    SET ( bestminorver ${CMAKE_MATCH_2})
    Message(STATUS "found version ${bestver}")
	endif()
  else()
    Message(WARNING "Could NOT extract the version from OptiX folder : ${result}")
  endif()
  find_path( OPTIX_INCLUDE_DIR optix.h ${OPTIX_LOCATION}/include )
  if ( OPTIX_INCLUDE_DIR )
    set (OPTIX_SDK_DIR ${OPTIX_INCLUDE_DIR}/../ )
  endif()
endif()
if(NOT DEFINED OPTIX_SDK_DIR)
 # Locate OptiX by version
 set ( SEARCH_PATHS
  $ENV{OPTIX_LOCATION}
  ${OPTIX_LOCATION}
  ${PROJECT_SOURCE_DIR}/../LocalPackages/Optix
  ${PROJECT_SOURCE_DIR}/../../LocalPackages/Optix
  ${PROJECT_SOURCE_DIR}/../../../LocalPackages/Optix
  C:/ProgramData/NVIDIA\ Corporation

 )
 
 _find_version_path ( OPTIX_VERSION OPTIX_SDK_DIR "OptiX" "${SEARCH_PATHS}" )
 
 message ( STATUS "OptiX version: ${OPTIX_VERSION}")
endif()

if (OPTIX_SDK_DIR)
  #-------- Locate HEADERS
  _find_files( OPTIX_HEADERS OPTIX_SDK_DIR "optix.h" "optix.h" "include/" )
  if(OPTIX_INCLUDE_DIR)
    set( OPTIX_FOUND "YES" )      
  endif()
else(OPTIX_SDK_DIR)
  message(WARNING "
      OPTIX not found. 
      The OPTIX folder you would specify with OPTIX_LOCATION should contain:
      - include folder: containing the include files"
  )
endif(OPTIX_SDK_DIR)

include(FindPackageHandleStandardArgs)

SET(OPTIX_INCLUDE_DIR "${OPTIX_SDK_DIR}/include" CACHE PATH "path")
add_definitions("-DOPTIX_PATH=R\"(${OPTIX_SDK_DIR})\"")
add_definitions("-DOPTIX_VERSION_STR=\"${OPTIX_VERSION}\"")

find_package_handle_standard_args(OPTIX
	REQUIRED_VARS OPTIX_INCLUDE_DIR
	VERSION_VAR OPTIX_VERSION
)

mark_as_advanced( OPTIX_FOUND )


