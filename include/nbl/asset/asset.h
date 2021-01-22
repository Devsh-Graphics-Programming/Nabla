// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_H_INCLUDED__
#define __NBL_ASSET_H_INCLUDED__

#include "nbl/asset/compile_config.h"

// dependencies
#include "nbl/system/system.h"

// utils
#include "nbl/asset/asset_utils.h"

// format
#include "nbl/asset/format/EFormat.h"
#include "nbl/asset/format/convertColor.h"
#include "nbl/asset/format/decodePixels.h"
#include "nbl/asset/format/encodePixels.h"

// base
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/IMesh.h"

// images
#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/ICPUImageView.h"
// filters
#include "nbl/asset/filters/CFillImageFilter.h"
#include "nbl/asset/filters/CCopyImageFilter.h"
#include "nbl/asset/filters/CPaddedCopyImageFilter.h"
#include "nbl/asset/filters/CConvertFormatImageFilter.h"
#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"
#include "nbl/asset/filters/CFlattenRegionsImageFilter.h"
#include "nbl/asset/filters/CMipMapGenerationImageFilter.h"

// shaders
#include "nbl/asset/ShaderRes.h"
#include "nbl/asset/IIncluder.h"
#include "nbl/asset/IIncludeHandler.h"
#include "nbl/asset/IBuiltinIncludeLoader.h"
#include "nbl/asset/IGLSLCompiler.h"
#include "nbl/asset/ISPIR_VProgram.h"
#include "nbl/asset/ICPUShader.h"
#include "nbl/asset/ICPUSpecializedShader.h"
#include "nbl/asset/CShaderIntrospector.h"
// pipelines

// meshes
#include "nbl/asset/ICPUMeshBuffer.h"
#include "nbl/asset/ICPUMesh.h"
#include "nbl/asset/CCPUMesh.h" // refactor
#include "nbl/asset/IGeometryCreator.h"
#include "nbl/asset/IMeshPacker.h"

// manipulation + reflection + introspection
#include "nbl/asset/IMeshManipulator.h"

// baw files
#include "nbl/asset/bawformat/CBAWFile.h"
#include "nbl/asset/bawformat/CBlobsLoadingManager.h"


// importexport
#include "nbl/asset/IAssetLoader.h"
#include "nbl/asset/IAssetManager.h"
#include "nbl/asset/IAssetWriter.h"
#include "nbl/asset/COpenEXRImageMetadata.h"
#include "nbl/asset/CMTLPipelineMetadata.h"
#include "nbl/asset/CPLYPipelineMetadata.h"
#include "nbl/asset/CSTLPipelineMetadata.h"

//VT
#include "nbl/asset/IVirtualTexture.h"
#include "nbl/asset/ICPUVirtualTexture.h"

#endif
