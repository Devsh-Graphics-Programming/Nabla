#ifndef _IRR_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_
#define _IRR_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_

#include "irrlicht.h"

namespace irr
{
namespace ext
{
namespace FullScreenTriangle
{


#include "irr/irrpack.h"
struct ScreenTriangleVertexStruct
{
    int16_t     Pos[2];
    uint16_t    TexCoord[2];
} PACK_STRUCT;
#include "irr/irrunpack.h"


inline video::IGPUMeshBuffer* createFullScreenTriangle(video::IVideoDriver* driver)
{
    ScreenTriangleVertexStruct vertices[3];
    vertices[0] = {{-1,-1},{0,0}};
    vertices[1] = {{ 3,-1},{2,0}};
    vertices[2] = {{-1, 3},{0,2}};

    video::IGPUBuffer* buff = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(vertices));
    auto upStreamBuff = driver->getDefaultUpStreamingBuffer();
    uint32_t offset = video::StreamingTransientDataBufferMT<>::invalid_address;
    const uint32_t size = sizeof(vertices);
    // multi_place can fail to allocate if the memory has not been freed yet and it times out on the wait (see comment to multi_free)_
    while (offset==video::StreamingTransientDataBufferMT<>::invalid_address)
    {
        // attribute start offsets need to be aligned to mult. of 4
        uint32_t alignment = 4u;
        const void* data = vertices;
        upStreamBuff->multi_place(std::chrono::seconds(1u),1u,&data,&offset,&size,&alignment);
    }
    // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
    if (upStreamBuff->needsManualFlushOrInvalidate())
        driver->flushMappedMemoryRanges({{upStreamBuff->getBuffer()->getBoundMemory(),offset,size}});
    // after we make sure writes are in GPU memory (visible to GPU) and not still in a cache, we can copy using the GPU to device-only memory
    driver->copyBuffer(upStreamBuff->getBuffer(),buff,offset,0,size);
    // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
    upStreamBuff->multi_free(1u,&offset,&size,driver->placeFence());

    video::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();

    video::IGPUMeshBuffer* triangleMeshBuffer = new video::IGPUMeshBuffer();
    triangleMeshBuffer->setMeshDataAndFormat(desc);
    desc->drop();

    desc->setVertexAttrBuffer(buff,asset::EVAI_ATTR0,asset::EF_R16G16_SSCALED,sizeof(ScreenTriangleVertexStruct),0);
    desc->setVertexAttrBuffer(buff,asset::EVAI_ATTR1,asset::EF_R16G16_USCALED,sizeof(ScreenTriangleVertexStruct),offsetof(ScreenTriangleVertexStruct,TexCoord[0]));
    desc->setIndexBuffer(buff);
    triangleMeshBuffer->setIndexCount(3u);
    buff->drop();

    return triangleMeshBuffer;
}

}
}
}

#endif

