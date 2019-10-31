#ifndef __IRR_PCH_ASSET_H_INCLUDED__
#define __IRR_PCH_ASSET_H_INCLUDED__

#include "irr/asset/asset.h"

// private headers (would be useful to cleanup the folders a bit)

	// buffers
		// loaders
		#include "irr/asset/CBufferLoaderBIN.h"

	// image
		// loaders
		#include "irr/asset/CImageLoaderDDS.h"
		#include "irr/asset/CImageLoaderJPG.h"
		#include "irr/asset/CImageLoaderPNG.h"
		#include "irr/asset/CImageLoaderTGA.h"
		// writers
		#include "irr/asset/CImageWriterJPG.h"
		#include "irr/asset/CImageWriterPNG.h"
		#include "irr/asset/CImageWriterTGA.h"

	// shaders
	#include "irr/asset/CShaderIntrospector.h"
	#include "irr/asset/CIncludeHandler.h"
	#include "irr/asset/CBuiltinIncluder.h"
	#include "irr/asset/CFilesystemIncluder.h"
		// builtins/headers
		#include "irr/asset/CGLSLScanBuiltinIncludeLoader.h"
		#include "irr/asset/CGLSLSkinningBuiltinIncludeLoader.h"

	
	// mesh
	#include "irr/asset/CGeometryCreator.h"
		// loaders
		#include "irr/asset/COBJMeshFileLoader.h"
		#include "irr/asset/CPLYMeshFileLoader.h"
		#include "irr/asset/CSTLMeshFileLoader.h"
		#include "irr/asset/CXMeshFileLoader.h"
		// writers
		#include "irr/asset/CPLYMeshWriter.h"
		#include "irr/asset/CSTLMeshWriter.h"
		// manipulation
		#include "irr/asset/CForsythVertexCacheOptimizer.h"
		#include "irr/asset/CSmoothNormalGenerator.h"
		#include "irr/asset/COverdrawMeshOptimizer.h"
		#include "irr/asset/CMeshManipulator.h"

	// baw file format
	#include "irr/asset/bawformat/legacy/CBAWLegacy.h"
	#include "irr/asset/CBAWMeshFileLoader.h"
	#include "irr/asset/CBAWMeshWriter.h"

#endif