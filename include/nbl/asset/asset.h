// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_H_INCLUDED__
#define __NBL_ASSET_H_INCLUDED__

#include "nbl/asset/compile_config.h"

// dependencies
#include "nbl/system/declarations.h"
#include "nbl/system/definitions.h" // TODO: split `asset.h` into decl and def

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
#include "nbl/asset/ISPIR_VProgram.h"
#include "nbl/asset/ICPUShader.h"
#include "nbl/asset/ICPUSpecializedShader.h"
#include "nbl/asset/utils/ShaderRes.h"
#include "nbl/asset/utils/IIncluder.h"
#include "nbl/asset/utils/IIncludeHandler.h"
#include "nbl/asset/utils/IBuiltinIncludeLoader.h"
#include "nbl/asset/utils/IGLSLCompiler.h"
#include "nbl/asset/utils/CShaderIntrospector.h"

// pipelines

// skinning
#include "nbl/asset/ICPUAnimationLibrary.h"
#include "nbl/asset/ICPUSkeleton.h"

// meshes
#include "nbl/asset/ICPUMeshBuffer.h"
#include "nbl/asset/ICPUMesh.h"
#include "nbl/asset/utils/IGeometryCreator.h"
#include "nbl/asset/utils/IMeshPacker.h"

// manipulation + reflection + introspection
#include "nbl/asset/utils/IMeshManipulator.h"

// baw files
#include "nbl/asset/bawformat/CBAWFile.h"
#include "nbl/asset/bawformat/CBlobsLoadingManager.h"


#include "nbl/asset/IAssetManager.h"
// importexport
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/interchange/IImageLoader.h"
#include "nbl/asset/interchange/IRenderpassIndependentPipelineLoader.h"
#include "nbl/asset/interchange/IAssetWriter.h"
#include "nbl/asset/interchange/IImageWriter.h"
#include "nbl/asset/metadata/COpenEXRMetadata.h"
#include "nbl/asset/metadata/CMTLMetadata.h"
#include "nbl/asset/metadata/COBJMetadata.h"
#include "nbl/asset/metadata/CPLYMetadata.h"
#include "nbl/asset/metadata/CSTLMetadata.h"

//VT
#include "nbl/asset/utils/IVirtualTexture.h"
#include "nbl/asset/utils/ICPUVirtualTexture.h"

#endif
