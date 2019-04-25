#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <iostream>
#include <cstdio>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"


using namespace irr;
using namespace core;


/**
We do cool stuff here, like make an event receiver to process input
**/
class MyEventReceiver : public IEventReceiver
{
public:

	MyEventReceiver()
	{
	}

	bool OnEvent(const SEvent& event)
	{
        if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
        {
            switch (event.KeyInput.Key)
            {
            case irr::KEY_KEY_Q: // switch wire frame mode
                exit(0);
                return true;
            default:
                break;
            }
        }

		return false;
	}

private:
};

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
public:
    SimpleCallBack() : mvpUniformLocation(-1), mvpUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        //! Normally we'd iterate through the array and check our actual constant names before mapping them to locations but oh well
        mvpUniformLocation = constants[0].location;
        mvpUniformType = constants[0].type;
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);
    }

    virtual void OnUnsetMaterial() {}
};

#include "irr/irrpack.h"
struct VertexStruct
{
    /// every member needs to be at location aligned to its type size for GLSL
    float Pos[3]; /// uses float hence need 4 byte alignment
    uint8_t Col[2]; /// same logic needs 1 byte alignment
    uint8_t uselessPadding[2]; /// so if there is a member with 4 byte alignment then whole struct needs 4 byte align, so pad it
} PACK_STRUCT;
#include "irr/irrunpack.h"

int main()
{
	printf("Please select the background:\n");
	printf(" (0 : default) Use SkyDome\n");
	printf(" (1) Use SkyBox\n");

	uint8_t c = 0;
	std::cin >> c;

	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();
    SimpleCallBack* callBack = new SimpleCallBack();

    //! First need to make a material other than default to be able to draw with custom shader
    video::SGPUMaterial material;
    material.BackfaceCulling = false; //! Triangles will be visible from both sides
    material.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        callBack,
                                                        0); //! No custom user data
    callBack->drop();


	scene::ISceneManager* smgr = device->getSceneManager();




	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	driver->setTextureCreationFlag(video::ETCF_CREATE_MIP_MAPS, false);

	// create skybox and skydome
	switch (c)
	{
        case '1':
        {
            asset::IAssetManager& assetMgr = device->getAssetManager();
            asset::IAssetLoader::SAssetLoadParams lparams;
            asset::ICPUTexture* cputextures[] {
                static_cast<asset::ICPUTexture*>(assetMgr.getAsset("../../media/irrlicht2_up.jpg", lparams)),
                static_cast<asset::ICPUTexture*>(assetMgr.getAsset("../../media/irrlicht2_dn.jpg", lparams)),
                static_cast<asset::ICPUTexture*>(assetMgr.getAsset("../../media/irrlicht2_lf.jpg", lparams)),
                static_cast<asset::ICPUTexture*>(assetMgr.getAsset("../../media/irrlicht2_rt.jpg", lparams)),
                static_cast<asset::ICPUTexture*>(assetMgr.getAsset("../../media/irrlicht2_ft.jpg", lparams)),
                static_cast<asset::ICPUTexture*>(assetMgr.getAsset("../../media/irrlicht2_bk.jpg", lparams))
            };
            core::vector<video::ITexture*> gputextures = driver->getGPUObjectsFromAssets(cputextures, cputextures+6);

            smgr->addSkyBoxSceneNode(
                gputextures[0],
                gputextures[1],
                gputextures[2],
                gputextures[3],
                gputextures[4],
                gputextures[5]
            );
        }
            break;
        default:
        {
            asset::IAssetLoader::SAssetLoadParams lparams;
            asset::ICPUTexture* cputexture = static_cast<asset::ICPUTexture*>(device->getAssetManager().getAsset("../../media/skydome.jpg", lparams));
            video::ITexture* skydomeTexture = driver->getGPUObjectsFromAssets(&cputexture, (&cputexture)+1).front();
            smgr->addSkyDomeSceneNode(skydomeTexture,16,8,0.95f,2.0f,10.f);

        }
            break;
	}

	driver->setTextureCreationFlag(video::ETCF_CREATE_MIP_MAPS, true);


	//! we want to move around the scene and view it from different angles
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.001f);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(10.0f);

    smgr->setActiveCamera(camera);


	//! disable mouse cursor, since camera will force it to the middle
	//! and we don't want a jittery cursor in the middle distracting us
	device->getCursorControl()->setVisible(false);

	//! Since our cursor will be enslaved, there will be no way to close the window
	//! So we listen for the "Q" key being pressed and exit the application
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

	uint64_t lastFPSTime = 0;

	while(device->run())
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();

        //! Stress test for memleaks aside from demo how to create meshes that live on the GPU RAM
        {
            VertexStruct vertices[8];
            vertices[0] = VertexStruct{{-1.f,-1.f,-1.f},{  0,  0}};
            vertices[1] = VertexStruct{{ 1.f,-1.f,-1.f},{127,  0}};
            vertices[2] = VertexStruct{{-1.f, 1.f,-1.f},{255,  0}};
            vertices[3] = VertexStruct{{ 1.f, 1.f,-1.f},{  0,127}};
            vertices[4] = VertexStruct{{-1.f,-1.f, 1.f},{127,127}};
            vertices[5] = VertexStruct{{ 1.f,-1.f, 1.f},{255,127}};
            vertices[6] = VertexStruct{{-1.f, 1.f, 1.f},{  0,255}};
            vertices[7] = VertexStruct{{ 1.f, 1.f, 1.f},{127,255}};

            uint16_t indices_indexed16[] =
            {
                0,1,2,1,2,3,
                4,5,6,5,6,7,
                0,1,4,1,4,5,
                2,3,6,3,6,7,
                0,2,4,2,4,6,
                1,3,5,3,5,7
            };

            auto upStreamBuff = driver->getDefaultUpStreamingBuffer();
            const void* dataToPlace[2]  = {vertices,indices_indexed16};
            uint32_t offsets[2]         = {video::StreamingTransientDataBufferMT<>::invalid_address,video::StreamingTransientDataBufferMT<>::invalid_address};
            uint32_t alignments[2]      = {sizeof(decltype(vertices[0u])),sizeof(decltype(indices_indexed16[0u]))};
            uint32_t sizes[2]           = {sizeof(vertices),sizeof(indices_indexed16)};
            upStreamBuff->multi_place(2u,(const void* const*)dataToPlace,(uint32_t*)offsets,(uint32_t*)sizes,(uint32_t*)alignments);
            if (upStreamBuff->needsManualFlushOrInvalidate())
            {
                auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
                driver->flushMappedMemoryRanges({video::IDriverMemoryAllocation::MappedMemoryRange(upStreamMem,offsets[0],sizes[0]),video::IDriverMemoryAllocation::MappedMemoryRange(upStreamMem,offsets[1],sizes[1])});
            }


            auto buff = upStreamBuff->getBuffer();
            video::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();
            desc->setVertexAttrBuffer(buff,asset::EVAI_ATTR0,asset::EF_R32G32B32_SFLOAT,sizeof(VertexStruct),offsetof(VertexStruct,Pos[0])+offsets[0]);
            desc->setVertexAttrBuffer(buff,asset::EVAI_ATTR1,asset::EF_R8G8_UNORM,sizeof(VertexStruct),offsetof(VertexStruct,Col[0])+offsets[0]);
            desc->setIndexBuffer(buff);


            video::IGPUMeshBuffer* mb = new video::IGPUMeshBuffer();
            mb->setMeshDataAndFormat(desc);
            mb->setIndexBufferOffset(offsets[1]);
            mb->setIndexType(asset::EIT_16BIT);
            mb->setIndexCount(2*3*6);
            desc->drop();


            driver->setTransform(video::E4X3TS_WORLD,core::matrix4x3());
            driver->setMaterial(material);
            driver->drawMeshBuffer(mb);
            mb->drop();

            upStreamBuff->multi_free(2u,(uint32_t*)&offsets,(uint32_t*)&sizes,driver->placeFence());
        }

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"GPU Mesh Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

    //create a screenshot
	video::IImage* screenshot = driver->createImage(asset::EF_B8G8R8A8_UNORM,params.WindowSize);
    glReadPixels(0,0, params.WindowSize.Width,params.WindowSize.Height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, screenshot->getData());
    {
        // images are horizontally flipped, so we have to fix that here.
        uint8_t* pixels = (uint8_t*)screenshot->getData();

        const int32_t pitch=screenshot->getPitch();
        uint8_t* p2 = pixels + (params.WindowSize.Height - 1) * pitch;
        uint8_t* tmpBuffer = new uint8_t[pitch];
        for (uint32_t i=0; i < params.WindowSize.Height; i += 2)
        {
            memcpy(tmpBuffer, pixels, pitch);
            memcpy(pixels, p2, pitch);
            memcpy(p2, tmpBuffer, pitch);
            pixels += pitch;
            p2 -= pitch;
        }
        delete [] tmpBuffer;
    }
    asset::CImageData* img = new asset::CImageData(screenshot);
    asset::IAssetWriter::SAssetWriteParams wparams(img);
    device->getAssetManager().writeAsset("screenshot.png", wparams);
    img->drop();
    screenshot->drop();

	device->drop();

	return 0;
}
