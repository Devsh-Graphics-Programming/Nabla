/* GCC compile Flags
-flto
-fuse-linker-plugin
-fno-omit-frame-pointer //for debug
-msse3
-mfpmath=sse
-ggdb3 //for debug
*/
/* Linker Flags
-lIrrlicht
-lXrandr
-lGL
-lX11
-lpthread
-ldl

-fuse-ld=gold
-flto
-fuse-linker-plugin
-msse3
*/
#include <irrlicht.h>
#include "driverChoice.h"

/**
This example shows how to:
1) Set up and Use a Simple Shader
2) render triangle buffers to screen in all the different ways
**/
using namespace irr;
using namespace core;


/*
The start of the main function starts like in most other example. We ask the
user for the desired renderer and start it up.
*/
int main()
{
	printf("Please select the mesh setup method you want:\n");
	printf(" (0 : default) Share Color Data Buffer Between Mesh Buffers\n");
	printf(" (1) Use Separate Color Data Buffers For Each Mesh Buffers\n");

	char c;
	std::cin >> c;

	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
    params.AntiAlias = 0; //No AA, yet
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<u32>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();

    //! First need to make a material other than default to be able to draw with custom shader
    video::SMaterial material;
    //material.BackfaceCulling = false; //! Triangles will be visible from both sides
    material.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../triangle.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../triangle.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        NULL, //! No Shader Callback (we dont have any constants/uniforms to pass to the shader)
                                                        0); //! No custom user data



	scene::ISceneManager* smgr = device->getSceneManager();



    //! A Triangle has 3 vertices
    const size_t triangleNumVerts = 3;
    /**
    Normalized Device Coordinates
    X,Y: [-1,1]
    Z (Depth): [-1.1] in OGL

    ===============================================================================================================================================
                                                        A WORD OF WARNING
                                                        A WORD OF WARNING
                                                        A WORD OF WARNING
                                                        A WORD OF WARNING
                                                        A WORD OF WARNING
    ===============================================================================================================================================

    Not all combinations of per-vertex component numbers and component data types are valid,
    the IMeshBuffer::mapVertexAttrBuffer function will fail (just return) in these cases
    Use bool scene::validCombination(const E_COMPONENT_TYPE& type, const E_COMPONENTS_PER_ATTRIBUTE& components) to check combinations
    **/
    float triangleVertPositions[] = {
        //random rubbish at the start, to test offset mechanism
        FLT_MAX,FLT_MIN,FLT_RADIX,
        12.f,34.f,-5.f,
        //first triangle data
        -1.f,-1.f,0.5f,
        1.f,-1.f,0.5f,
        0.f,1.f,0.5f,
        //second triangle data
        -1.f,1.f,0.8f,
        1.f,1.f,0.8f,
        0.f,-1.f,0.8f
    };
    //! We can demonstrate the usage of arbitrary precision vertex attributes
    //! In this case we use HDR, which will be outside of the screen's displayable range of 24bits [0.f to 1.f]
    //! So Colors in the Corners will appear oversaturated
    //! Also this exotic layout has no alpha per vertex!
    float triangleVertColors_1[] = {
        //no rubbish at the start
        //first triangle data
        2.f,-1.f,-1.f,
        -1.f,2.f,-1.f,
        -1.f,-1.f,2.f
    };
    //! These are normal/usual/old irrlicht color ranges for vertices
    //! Using BGRA byte order which requires special component flag
    uint8_t triangleVertColors_2[] = {
        255,0,0,255,//blue
        255,0,0,255,//blue
        0,0,255,255//red
    };
    /** CPU-Side objects such as buffers are drop()'ped to delete
        Aside from demonstrating precision flexibility,
        the choice of 2 different attribute formats
        forces us to render the two triangles in separate drawcalls.
        Therefore to demonstrate the flexibility of MeshBuffer and IBuffer mapping.
    **/
    scene::ICPUMeshBuffer* mb_1 = new scene::ICPUMeshBuffer();
    scene::ICPUMeshBuffer* mb_2 = new scene::ICPUMeshBuffer();
    scene::ICPUMeshDataFormatDesc* desc_1 = new scene::ICPUMeshDataFormatDesc();
    mb_1->setMeshDataAndFormat(desc_1);
    desc_1->drop();
    scene::ICPUMeshDataFormatDesc* desc_2 = new scene::ICPUMeshDataFormatDesc();
    mb_2->setMeshDataAndFormat(desc_2);
    desc_2->drop();


    //! There are 3 position components per vertex, each a float
    ICPUBuffer* positionBuf = new ICPUBuffer(sizeof(triangleVertPositions));
    //! Creating a CPU buffer is decoupled from filling it, so memcpy is needed
    memcpy(positionBuf->getPointer(),triangleVertPositions,positionBuf->getSize());


    //! By mapping we increase/grab() ref counter of positionBuf, any previously mapped buffer will have it's reference dropped
    desc_1->mapVertexAttrBuffer(positionBuf,
                            scene::EVAI_ATTR0, //! we use first attribute slot (out of a minimum of 16)
                            scene::ECPA_THREE, //! there are 3 components per vertex
                            scene::ECT_FLOAT, //! and they are floats
                            12, //! each per-vertex bundle of components is 12 bytes is this case
                            24); //! Offset of 24bytes from the start of the buffer
    //! When mapping a stride of 0 (default) tells irrlicht to compute the stride automatically from num of components and type
    desc_2->mapVertexAttrBuffer(positionBuf,scene::EVAI_ATTR0,scene::ECPA_THREE,scene::ECT_FLOAT,0,24+triangleNumVerts*4*3);

    /** Since we mapped the buffer, the MeshBuffers will be using it.
        If we drop it, it will be automatically deleted when MeshBuffers are done using it.
    **/
    positionBuf->drop();


    //! This is slightly more complicated
    ICPUBuffer* colorBuf;
    ICPUBuffer* colorBufExtra;
    switch (c)
    {
        //! One data buffer per meshbuffer
        case '1':
            colorBuf = new ICPUBuffer(sizeof(triangleVertColors_1));
            colorBufExtra = new ICPUBuffer(sizeof(triangleVertColors_2));
            //fill
            memcpy(colorBuf->getPointer(),triangleVertColors_1,sizeof(triangleVertColors_1));
            memcpy(colorBufExtra->getPointer(),triangleVertColors_2,sizeof(triangleVertColors_2));
            //! The offset into the buffer is 0 by default
            desc_1->mapVertexAttrBuffer(colorBuf,scene::EVAI_ATTR1,scene::ECPA_THREE,scene::ECT_FLOAT);
            //! The ECT_NORMALIZED_* types rescale the integer values into [-1.f,1.f] and [0.f,1.f] for signed and unsigned respectively
            //! ECPA_REVERSED_OR_BGRA is a special alias for ECPA_FOUR to indicate RGBA is going to be swizzled into BGRA
            desc_2->mapVertexAttrBuffer(colorBufExtra,scene::EVAI_ATTR1,scene::ECPA_REVERSED_OR_BGRA,scene::ECT_NORMALIZED_UNSIGNED_BYTE);
            //drop
            colorBuf->drop();
            colorBufExtra->drop();
            break;
        //! Sharing one buffer for 2 different meshbuffers
        default:
            colorBuf = new ICPUBuffer(sizeof(triangleVertColors_1)+sizeof(triangleVertColors_2));
            //fill
            memcpy(colorBuf->getPointer(),triangleVertColors_1,sizeof(triangleVertColors_1));
            //! casting to uint8_t has no significance here, just so we can do byte-size pointer math
            memcpy(((uint8_t*)colorBuf->getPointer())+sizeof(triangleVertColors_1),triangleVertColors_2,sizeof(triangleVertColors_2));
            //map first buffer
            desc_1->mapVertexAttrBuffer(colorBuf,scene::EVAI_ATTR1,scene::ECPA_THREE,scene::ECT_FLOAT);
            desc_2->mapVertexAttrBuffer(colorBuf,scene::EVAI_ATTR1,scene::ECPA_REVERSED_OR_BGRA,scene::ECT_NORMALIZED_UNSIGNED_BYTE,0,sizeof(triangleVertColors_1));
            //drop
            colorBuf->drop();
            break;
    }


    //! We will have to drop "mb" manually to delete it since nothing else grabs it.


    /**
    I'll explain the mapping of a index buffer later.

    If we dont use an index buffer, it means that each vertex is used only once.
    (as if the index buffer is simply {0,1,2,3,4,.....,N-1} and index count is N)

    We still need to tell it how many vertices to use (hence primitives to render)
    **/
    uint32_t indices[] = {2,1,0};
    ICPUBuffer* ixbuf = new ICPUBuffer(sizeof(indices));
    memcpy(ixbuf->getPointer(),indices,sizeof(indices));
    desc_1->mapIndexBuffer(ixbuf);
    mb_1->setIndexType(video::EIT_32BIT);
    mb_1->setIndexCount(3);
    ixbuf->drop();
    mb_2->setIndexCount(triangleNumVerts);

	uint64_t lastFPSTime = 0;

	while(device->run())
	if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

        driver->setTransform(video::ETS_WORLD,core::matrix4());
        driver->setTransform(video::ETS_VIEW,core::matrix4());
        driver->setTransform(video::ETS_PROJECTION,core::matrix4());
        driver->setMaterial(material);
        //! draw back to front
        driver->drawMeshBuffer(mb_2);
        driver->drawMeshBuffer(mb_1);

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			stringw str = L"Terrain Renderer - Irrlicht Engine [";
			str += driver->getName();
			str += "] FPS:";
			str += driver->getFPS();
			str += " PrimitvesDrawn:";
			str += driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.c_str());
			lastFPSTime = time;
		}
	}
	mb_1->drop();
	mb_2->drop();

	device->drop();

	return 0;
}
