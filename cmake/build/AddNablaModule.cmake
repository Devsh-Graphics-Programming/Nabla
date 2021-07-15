# Copyright (c) 2019 DevSH Graphics Programming Sp. z O.O.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(addNablaModule target NBL_INSTALL_DIR)	
	target_include_directories(${target}
    PUBLIC
        $<$<CONFIG:Debug>:${NBL_INSTALL_DIR}/debug/include>
        $<$<CONFIG:RelWithDebInfo>:${NBL_INSTALL_DIR}/relwithdebinfo/include>
        $<$<CONFIG:Release>:${NBL_INSTALL_DIR}/include>
        # these are needed because we haven't cleaned up the API properly yet
        $<$<CONFIG:Debug>:${NBL_INSTALL_DIR}/debug/source/Nabla>
        $<$<CONFIG:RelWithDebInfo>:${NBL_INSTALL_DIR}/relwithdebinfo/source/Nabla>
        $<$<CONFIG:Release>:${NBL_INSTALL_DIR}/source/Nabla>
	)
	target_link_libraries(${target} PRIVATE
		 $<$<CONFIG:Debug>:${NBL_INSTALL_DIR}/debug/lib/Nabla_debug.lib>
		 $<$<CONFIG:RelWithDebInfo>:${NBL_INSTALL_DIR}/relwithdebinfo/lib/Nabla_rwdi.lib>
		 $<$<CONFIG:Release>:${NBL_INSTALL_DIR}/lib/Nabla.lib>
	)
	
	function(link_nbl_dependency DEPENDENCY_NAME)
		target_link_libraries(${target} PRIVATE
			 $<$<CONFIG:Debug>:${NBL_INSTALL_DIR}/debug/lib/${DEPENDENCY_NAME}d.lib> # d POSTFIX
			 $<$<CONFIG:RelWithDebInfo>:${NBL_INSTALL_DIR}/relwithdebinfo/lib/${DEPENDENCY_NAME}.lib>
			 $<$<CONFIG:Release>:${NBL_INSTALL_DIR}/lib/${DEPENDENCY_NAME}.lib>
		)
	endfunction()
	
	function(link_nbl_dependency_ DEPENDENCY_NAME)
		target_link_libraries(${target} PRIVATE
			 $<$<CONFIG:Debug>:${NBL_INSTALL_DIR}/debug/lib/${DEPENDENCY_NAME}_d.lib> # _d POSTFIX
			 $<$<CONFIG:RelWithDebInfo>:${NBL_INSTALL_DIR}/relwithdebinfo/lib/${DEPENDENCY_NAME}.lib>
			 $<$<CONFIG:Release>:${NBL_INSTALL_DIR}/lib/${DEPENDENCY_NAME}.lib>
		)
	endfunction()
	
	function(link_nbl_dependency_nond DEPENDENCY_NAME)
		target_link_libraries(${target} PRIVATE
			 $<$<CONFIG:Debug>:${NBL_INSTALL_DIR}/debug/lib/${DEPENDENCY_NAME}.lib> # none d POSTFIX
			 $<$<CONFIG:RelWithDebInfo>:${NBL_INSTALL_DIR}/relwithdebinfo/lib/${DEPENDENCY_NAME}.lib>
			 $<$<CONFIG:Release>:${NBL_INSTALL_DIR}/lib/${DEPENDENCY_NAME}.lib>
		)
	endfunction()
	
	link_nbl_dependency(glslang)
	link_nbl_dependency_nond(jpeg)
	link_nbl_dependency(libpng16_static)
	
	# OpenSSL only ever exists in the Release variant
	if(WIN32)
		target_link_libraries(${target} PRIVATE
			 ${NBL_INSTALL_DIR}/lib/libeay32.lib
			 ${NBL_INSTALL_DIR}/lib/ssleay32.lib
		)
	else()
		target_link_libraries(${target} PRIVATE
			 ${NBL_INSTALL_DIR}/lib/libcrypto.a
			 ${NBL_INSTALL_DIR}/lib/libssl.a
		)
	endif()
	link_nbl_dependency(libpng16_static)
	link_nbl_dependency(OGLCompiler)
	link_nbl_dependency(OSDependent)
	link_nbl_dependency_nond(shaderc)
	link_nbl_dependency_nond(shaderc_util)
	link_nbl_dependency(SPIRV)
	link_nbl_dependency_nond(SPIRV-Tools)
	link_nbl_dependency_nond(SPIRV-Tools-opt)
	link_nbl_dependency(zlibstatic)
	link_nbl_dependency(GenericCodeGen)
	link_nbl_dependency(HLSL)
	link_nbl_dependency(MachineIndependent)
	link_nbl_dependency_nond(egl)
endfunction()