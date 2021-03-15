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
		#include "nbl/asset/CBufferLoaderBIN.h"

	// image
		// loaders
		#include "nbl/asset/CImageLoaderJPG.h"
		#include "nbl/asset/CImageLoaderPNG.h"
		#include "nbl/asset/CImageLoaderTGA.h"
		#include "nbl/asset/CImageLoaderOpenEXR.h"
		#include "nbl/asset/CGLILoader.h"
		// writers
		#include "nbl/asset/CImageWriterJPG.h"
		#include "nbl/asset/CImageWriterPNG.h"
		#include "nbl/asset/CImageWriterTGA.h"
		#include "nbl/asset/CImageWriterOpenEXR.h"
		#include "nbl/asset/CGLIWriter.h"

	// shaders
	#include "nbl/asset/CShaderIntrospector.h"
	#include "nbl/asset/CIncludeHandler.h"
	#include "nbl/asset/CBuiltinIncluder.h"
	#include "nbl/asset/CFilesystemIncluder.h"

		// builtins/headers
		#include "nbl/asset/CGLSLVirtualTexturingBuiltinIncludeLoader.h"

	
	// mesh
	#include "nbl/asset/CGeometryCreator.h"
		// loaders
		#include "nbl/asset/COBJMeshFileLoader.h"
		#include "nbl/asset/CPLYMeshFileLoader.h"
		#include "nbl/asset/CSTLMeshFileLoader.h"
		// writers
		#include "nbl/asset/CPLYMeshWriter.h"
		#include "nbl/asset/CSTLMeshWriter.h"
		// manipulation
		#include "nbl/asset/CForsythVertexCacheOptimizer.h"
		#include "nbl/asset/CSmoothNormalGenerator.h"
		#include "nbl/asset/COverdrawMeshOptimizer.h"
		#include "nbl/asset/CMeshManipulator.h"

	// baw file format
	#include "nbl/asset/bawformat/legacy/CBAWLegacy.h"
	#include "nbl/asset/CBAWMeshFileLoader.h"
	#include "nbl/asset/CBAWMeshWriter.h"
#endif //_NBL_PCH_IGNORE_PRIVATE_HEADERS

#endif