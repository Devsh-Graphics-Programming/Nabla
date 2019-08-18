#ifndef __IRR_ASSET_H_INCLUDED__
#define __IRR_ASSET_H_INCLUDED__

// dependencies
#include "irr/system/system.h"


// format
#include "irr/asset/format/EFormat.h"
#include "irr/asset/format/convertColor.h"
#include "irr/asset/format/decodePixels.h"
#include "irr/asset/format/encodePixels.h"

//! move around in folders soon
// base
#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/IAsset.h"
#include "irr/asset/IMesh.h"
#include "irr/asset/IMeshManipulator.h"

// images
#include "irr/asset/CImageData.h"
#include "irr/asset/ICPUTexture.h"
// shaders
#include "irr/asset/ShaderCommons.h"
#include "irr/asset/ShaderRes.h"
#include "irr/asset/IIncluder.h"
#include "irr/asset/IIncludeHandler.h"
#include "irr/asset/IBuiltinIncludeLoader.h"
#include "irr/asset/IParsedShaderSource.h"
#include "irr/asset/IGLSLCompiler.h"
#include "irr/asset/ISPIR_VProgram.h"
#include "irr/asset/ICPUShader.h"
#include "irr/asset/ICPUSpecializedShader.h"
// meshes
#include "irr/asset/ICPUMeshBuffer.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"
#include "irr/asset/ICPUMesh.h"
#include "irr/asset/CCPUMesh.h" // refactor
#include "irr/asset/ICPUSkinnedMesh.h"
#include "irr/asset/CCPUSkinnedMesh.h" // refactor
#include "irr/asset/IGeometryCreator.h"
// pipelines

// baw files
#include "irr/asset/bawformat/CBAWFile.h"
#include "irr/asset/bawformat/CBlobsLoadingManager.h"


// importexport
#include "irr/asset/IAssetLoader.h"
#include "irr/asset/IAssetManager.h"
#include "irr/asset/IAssetWriter.h"

#endif