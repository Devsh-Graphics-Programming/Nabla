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
#include "nbl/video/alloc/StreamingGPUBufferAllocator.h"
#include "nbl/video/alloc/HostDeviceMirrorBufferAllocator.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"

// utilities
#include "nbl/video/utilities/ICommandPoolCache.h"
#include "nbl/video/utilities/CPropertyPool.h"
#include "nbl/video/utilities/IUtilities.h"
#include "nbl/video/utilities/IGPUObjectFromAssetConverter.h"


// platform and API specific stuff
#include "nbl/video/COpenGL_Connection.h"
#include "nbl/video/surface/ISurfaceGL.h"
#include "nbl/video/CVulkanConnection.h"

//VT
//#include "nbl/video/IGPUVirtualTexture.h"

#endif