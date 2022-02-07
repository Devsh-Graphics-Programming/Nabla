// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_H_INCLUDED__
#define __NBL_VIDEO_H_INCLUDED__

#include "nbl/video/compile_config.h"

// dependencies
#include "nbl/asset/asset.h"
#include "nbl/ui/ui.h"  // unsure yet

// alloc
#include "nbl/video/alloc/GPUMemoryAllocatorBase.h"
#include "nbl/video/alloc/HostDeviceMirrorBufferAllocator.h"
#include "nbl/video/alloc/SimpleGPUBufferAllocator.h"
#include "nbl/video/alloc/ResizableBufferingAllocator.h"
#include "nbl/video/alloc/StreamingGPUBufferAllocator.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"
#include "nbl/video/alloc/SubAllocatedDataBuffer.h"

// properties
#include "nbl/video/CPropertyPool.h"
#include "nbl/video/CPropertyPoolHandler.h"

// think about foler name for those
#include "nbl/video/asset_traits.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/IGPUMeshBuffer.h"
#include "nbl/video/IGPUMesh.h"
#include "nbl/video/IGPUObjectFromAssetConverter.h"

//VT
#include "nbl/video/CGPUMeshPackerV2.h"
#include "nbl/video/IGPUVirtualTexture.h"

#endif