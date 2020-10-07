// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_H_INCLUDED__
#define __NBL_VIDEO_H_INCLUDED__

#include "irr/video/compile_config.h"

// dependencies
#include "irr/asset/asset.h"
#include "irr/ui/ui.h" // unsure yet

// alloc
#include "irr/video/alloc/GPUMemoryAllocatorBase.h"
#include "irr/video/alloc/HostDeviceMirrorBufferAllocator.h"
#include "irr/video/alloc/SimpleGPUBufferAllocator.h"
#include "irr/video/alloc/ResizableBufferingAllocator.h"
#include "irr/video/alloc/StreamingGPUBufferAllocator.h"
#include "irr/video/alloc/StreamingTransientDataBuffer.h"
#include "irr/video/alloc/SubAllocatedDataBuffer.h"

// think about foler name for those
#include "irr/video/asset_traits.h"
#include "irr/video/IGPUShader.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "irr/video/IGPUMeshBuffer.h"
#include "irr/video/IGPUMesh.h"
#include "irr/video/IGPUSkinnedMesh.h"
#include "irr/video/CGPUMesh.h"
#include "irr/video/IGPUObjectFromAssetConverter.h"

// kill/refactor
#include "irr/video/CGPUSkinnedMesh.h"

//VT
#include "irr/video/IGPUVirtualTexture.h"

#endif