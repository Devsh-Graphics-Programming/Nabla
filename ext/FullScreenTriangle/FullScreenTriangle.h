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


inline core::smart_refctd_ptr<video::IGPUMeshBuffer> createFullScreenTriangle(video::IVideoDriver* driver)
{
    ScreenTriangleVertexStruct vertices[3];
    vertices[0] = {{-1, 1},{0,0}};
    vertices[1] = {{-1,-3},{0,2}};
    vertices[2] = {{ 3, 1},{2,0}};

    auto buff = core::smart_refctd_ptr<video::IGPUBuffer>(driver->createDeviceLocalGPUBufferOnDedMem(sizeof(vertices)),core::dont_grab);
    driver->updateBufferRangeViaStagingBuffer(buff.get(),0,sizeof(vertices),vertices);


    auto triangleMeshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>();
	{
		auto desc = driver->createGPUMeshDataFormatDesc();

		desc->setVertexAttrBuffer(core::smart_refctd_ptr(buff),asset::EVAI_ATTR0,asset::EF_R8G8B8A8_SSCALED,sizeof(ScreenTriangleVertexStruct),0);
		desc->setVertexAttrBuffer(core::smart_refctd_ptr(buff),asset::EVAI_ATTR1,asset::EF_R8G8B8A8_SSCALED,sizeof(ScreenTriangleVertexStruct),offsetof(ScreenTriangleVertexStruct,TexCoord[0]));
		triangleMeshBuffer->setIndexCount(3u);
    
		triangleMeshBuffer->setMeshDataAndFormat(std::move(desc));
	}

    return triangleMeshBuffer;
}

}
}
}

#endif

