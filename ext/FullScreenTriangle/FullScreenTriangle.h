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
    int8_t     Pos[2];
    int8_t    TexCoord[2];
} PACK_STRUCT;
#include "irr/irrunpack.h"


inline scene::IGPUMeshBuffer* createFullScreenTriangle(video::IVideoDriver* driver)
{
    ScreenTriangleVertexStruct vertices[3];
    vertices[0] = {{-1, 1},{0,0}};
    vertices[1] = {{-1,-3},{0,2}};
    vertices[2] = {{ 3, 1},{2,0}};

    video::IGPUBuffer* buff = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(vertices));
    driver->updateBufferRangeViaStagingBuffer(buff,0,sizeof(vertices),vertices);

    scene::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();

    scene::IGPUMeshBuffer* triangleMeshBuffer = new scene::IGPUMeshBuffer();
    triangleMeshBuffer->setMeshDataAndFormat(desc);
    desc->drop();

    desc->mapVertexAttrBuffer(buff,scene::EVAI_ATTR0,scene::ECPA_FOUR,scene::ECT_BYTE,sizeof(ScreenTriangleVertexStruct),0);
    desc->mapIndexBuffer(buff);
    triangleMeshBuffer->setIndexCount(3u);
    buff->drop();

    return triangleMeshBuffer;
}

}
}
}

#endif

