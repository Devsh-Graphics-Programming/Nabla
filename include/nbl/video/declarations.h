// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_DECLARATIONS_H_INCLUDED__
#define __NBL_VIDEO_DECLARATIONS_H_INCLUDED__


// dependencies
#include "nbl/asset/asset.h"
#include "nbl/ui/declarations.h"

// alloc
#include "nbl/video/alloc/StreamingGPUBufferAllocator.h"
#include "nbl/video/alloc/HostDeviceMirrorBufferAllocator.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"

// properties
#include "nbl/video/CPropertyPool.h"
#include "nbl/video/CPropertyPoolHandler.h"

// think about foler name for those
#include "nbl/video/IAPIConnection.h"
#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/asset_traits.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/IGPUMeshBuffer.h"
#include "nbl/video/IGPUMesh.h"
#include "nbl/video/IGPUQueue.h"
#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUObjectFromAssetConverter.h"

//VT
//#include "nbl/video/IGPUVirtualTexture.h"

#endif