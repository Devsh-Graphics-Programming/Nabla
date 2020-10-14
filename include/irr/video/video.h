#ifndef __IRR_VIDEO_H_INCLUDED__
#define __IRR_VIDEO_H_INCLUDED__

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

// properties
#include "irr/video/CPropertyPool.h"
#include "irr/video/CPropertyPoolHandler.h"

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