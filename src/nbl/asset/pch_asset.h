// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_PCH_ASSET_H_INCLUDED__
#define __NBL_ASSET_PCH_ASSET_H_INCLUDED__

#include "nbl/asset/asset.h"

// private headers (would be useful to cleanup the folders a bit)
#ifndef _NBL_PCH_IGNORE_PRIVATE_HEADERS

// buffers
// loaders
#include "nbl/asset/interchange/CBufferLoaderBIN.h"

// image
// loaders
#include "nbl/asset/interchange/CImageLoaderJPG.h"
#include "nbl/asset/interchange/CImageLoaderPNG.h"
#include "nbl/asset/interchange/CImageLoaderTGA.h"
#include "nbl/asset/interchange/CImageLoaderOpenEXR.h"
#include "nbl/asset/interchange/CGLILoader.h"
// writers
#include "nbl/asset/interchange/CImageWriterJPG.h"
#include "nbl/asset/interchange/CImageWriterPNG.h"
#include "nbl/asset/interchange/CImageWriterTGA.h"
#include "nbl/asset/interchange/CImageWriterOpenEXR.h"
#include "nbl/asset/interchange/CGLIWriter.h"

// shaders
#include "nbl/asset/utils/CSPIRVIntrospector.h"
#include "nbl/asset/utils/IShaderCompiler.h"

// builtins/headers
#include "nbl/asset/utils/CGLSLVirtualTexturingBuiltinIncludeGenerator.h"


// mesh
#include "nbl/asset/utils/CGeometryCreator.h"
// loaders
#include "nbl/asset/interchange/COBJMeshFileLoader.h"
#include "nbl/asset/interchange/CPLYMeshFileLoader.h"
#include "nbl/asset/interchange/CSTLMeshFileLoader.h"
// writers
#include "nbl/asset/interchange/CPLYMeshWriter.h"
#include "nbl/asset/interchange/CSTLMeshWriter.h"
// manipulation
#include "nbl/asset/utils/CForsythVertexCacheOptimizer.h"
#include "nbl/asset/utils/CSmoothNormalGenerator.h"
#include "nbl/asset/utils/COverdrawMeshOptimizer.h"
#include "nbl/asset/utils/CMeshManipulator.h"

// baw file format - not valid anymore
//#include "nbl/asset/bawformat/legacy/CBAWLegacy.h"
#ifdef _NBL_COMPILE_WITH_BAW_LOADER_
//#include "nbl/asset/bawformat/CBAWMeshFileLoader.h"
#endif
#ifdef _NBL_COMPILE_WITH_BAW_WRITER_
//#include "nbl/asset/bawformat/CBAWMeshWriter.h"
#endif

#endif //_NBL_PCH_IGNORE_PRIVATE_HEADERS

#endif
