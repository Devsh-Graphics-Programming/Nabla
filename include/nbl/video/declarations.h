// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_DECLARATIONS_H_INCLUDED__
#define __NBL_VIDEO_DECLARATIONS_H_INCLUDED__


// dependencies
#include "nbl/asset/asset.h"
#include "nbl/ui/declarations.h"

// core objects
#include "nbl/video/IAPIConnection.h"
#include "nbl/video/IPhysicalDevice.h"
//#include "nbl/video/asset_traits.h"

// alloc
#include "nbl/video/alloc/CStreamingBufferAllocator.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"

// platform and API specific stuff
#include "nbl/video/COpenGL_Connection.h"
#include "nbl/video/CVulkanConnection.h"
// [TODO] problem with this class is that it has an IThreadHandler templated on the OpenGL function table
// if I leave the function table undefined, it errors, and if I load in the function table in the include header, 
// I get "'gl*': redefinition; different linkage"
//#include "nbl/video/COpenGL_Swapchain.h"
#include "nbl/video/CVulkanSwapchain.h"
#include "nbl/video/surface/CSurfaceGL.h"
#include "nbl/video/surface/CSurfaceVulkan.h"

// CUDA
#include "nbl/video/CCUDAHandler.h"

// utilities
#include "nbl/video/utilities/CDumbPresentationOracle.h"
#include "nbl/video/utilities/ICommandPoolCache.h"
#include "nbl/video/utilities/CPropertyPool.h"
#include "nbl/video/utilities/CDrawIndirectAllocator.h"
#include "nbl/video/utilities/CSubpassKiln.h"
#include "nbl/video/utilities/IUtilities.h"
#include "nbl/video/utilities/IGPUObjectFromAssetConverter.h"

//VT
//#include "nbl/video/CGPUMeshPackerV2.h"
//#include "nbl/video/IGPUVirtualTexture.h"


#endif