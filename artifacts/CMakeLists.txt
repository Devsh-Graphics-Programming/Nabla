# Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

set(EXAMPLES_TESTS_PATH ${NBL_ROOT_PATH}/examples_tests)
set(NBL_RAYTRACEDAO_EX_NAME "22.RaytracedAO")
set(NBL_DENOISER_TONEMAPPER_EX_NAME "39.DenoiserTonemapper")

add_custom_target(pack_artifact_ditt
	COMMAND cmake -E echo "Archiving the build!"
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/bin/LowDiscrepancySequenceCache.bin ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/bin/LowDiscrepancySequenceCache.bin
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/bin/raytracedao.exe ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/bin/raytracedao.exe
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/closestHit.comp ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/closestHit.comp
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/common.h ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/common.h
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/cull.comp ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/cull.comp
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/denoiser_hook.bat ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/denoiser_hook.bat
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/extractCubemap.bat ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/extractCubemap.bat
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/fillVisBuffer.frag ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/fillVisBuffer.frag
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/fillVisBuffer.vert ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/fillVisBuffer.vert
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/main.cpp ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/main.cpp
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/mergeCubemap.bat ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/mergeCubemap.bat
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/README.md ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/README.md
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/rasterizationCommon.h ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/rasterizationCommon.h
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/raygen.comp ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/raygen.comp
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/raytraceCommon.glsl ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/raytraceCommon.glsl
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/raytraceCommon.h ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/raytraceCommon.h
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/resolve.comp ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/resolve.comp
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/virtualGeometry.glsl ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/virtualGeometry.glsl
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_RAYTRACEDAO_EX_NAME}/virtualGeometry.glsl ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_RAYTRACEDAO_EX_NAME}/virtualGeometry.glsl

	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_DENOISER_TONEMAPPER_EX_NAME}/bin/denoisertonemapper.exe ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_DENOISER_TONEMAPPER_EX_NAME}/bin/denoisertonemapper.exe
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_DENOISER_TONEMAPPER_EX_NAME}/CommonPushConstants.h ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_DENOISER_TONEMAPPER_EX_NAME}/CommonPushConstants.h
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/${NBL_DENOISER_TONEMAPPER_EX_NAME}/ShaderCommon.glsl ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/${NBL_DENOISER_TONEMAPPER_EX_NAME}/ShaderCommon.glsl
	
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/media/blueNoiseDithering/LDR_RGBA.png ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/media/blueNoiseDithering/LDR_RGBA.png
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/media/kernels/physical_flare_256.exr ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/media/kernels/physical_flare_256.exr
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/media/kernels/physical_flare_512.exr ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/media/kernels/physical_flare_512.exr
	
	COMMAND cmake -E copy ${EXAMPLES_TESTS_PATH}/media/mitsuba/staircase2.zip ${CMAKE_CURRENT_BINARY_DIR}/Ditt/pack/media/mitsuba/staircase2.zip
	
	COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/Ditt && cmake -E tar -cvj Ditt.tar.bz2 pack/
)
