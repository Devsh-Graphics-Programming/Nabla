#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>
#include "../common/QToQuitEventReceiver.h"
#include "../../ext/ScreenShot/ScreenShot.h"


using namespace irr;
using namespace core;


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


	auto* assetMgr = device->getAssetManager();
	scene::ISceneManager* smgr = device->getSceneManager();




	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	driver->setTextureCreationFlag(video::ETCF_CREATE_MIP_MAPS, false);

	// create skybox and skydome
	switch (c)
	{
        case '1':
			{
				asset::IAssetLoader::SAssetLoadParams lparams;
				asset::ICPUTexture* cputextures[] {
					static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/irrlicht2_up.jpg", lparams).getContents().first->get()),
					static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/irrlicht2_dn.jpg", lparams).getContents().first->get()),
					static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/irrlicht2_lf.jpg", lparams).getContents().first->get()),
					static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/irrlicht2_rt.jpg", lparams).getContents().first->get()),
					static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/irrlicht2_ft.jpg", lparams).getContents().first->get()),
					static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/irrlicht2_bk.jpg", lparams).getContents().first->get())
				};
				auto gputextures = driver->getGPUObjectsFromAssets(cputextures, cputextures+6);

				smgr->addSkyBoxSceneNode(
					std::move(gputextures[0]),
					std::move(gputextures[1]),
					std::move(gputextures[2]),
					std::move(gputextures[3]),
					std::move(gputextures[4]), 
					std::move(gputextures[5])
				);
			}
            break;
        default:
			{
				asset::IAssetLoader::SAssetLoadParams lparams;
				auto cputexture = core::smart_refctd_ptr_static_cast<asset::ICPUTexture>(*assetMgr->getAsset("../../media/skydome.jpg", lparams).getContents().first);
				auto skydomeTexture = driver->getGPUObjectsFromAssets(&cputexture.get(), (&cputexture.get())+1).front();
				smgr->addSkyDomeSceneNode(std::move(skydomeTexture),16,8,0.95f,2.0f,10.f);

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
	QToQuitEventReceiver receiver;
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

			auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>();
			{
				auto desc = driver->createGPUMeshDataFormatDesc();
				{
					auto buff = core::smart_refctd_ptr<video::IGPUBuffer>(upStreamBuff->getBuffer());
					desc->setVertexAttrBuffer(core::smart_refctd_ptr(buff),asset::EVAI_ATTR0,asset::EF_R32G32B32_SFLOAT,sizeof(VertexStruct),offsetof(VertexStruct,Pos[0])+offsets[0]);
					desc->setVertexAttrBuffer(core::smart_refctd_ptr(buff),asset::EVAI_ATTR1,asset::EF_R8G8_UNORM,sizeof(VertexStruct),offsetof(VertexStruct,Col[0])+offsets[0]);
					desc->setIndexBuffer(std::move(buff));
				}

				mb->setIndexBufferOffset(offsets[1]);
				mb->setIndexType(asset::EIT_16BIT);
				mb->setIndexCount(2*3*6);
				mb->setMeshDataAndFormat(std::move(desc));
			}


            driver->setTransform(video::E4X3TS_WORLD,core::matrix4x3());
            driver->setMaterial(material);
            driver->drawMeshBuffer(mb.get());

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
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	device->drop();

	return 0;
}
