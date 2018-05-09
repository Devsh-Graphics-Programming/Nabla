#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <iostream>
#include <cstdio>

#include "../source/Irrlicht/COpenGLExtensionHandler.h"
#include "COpenGLStateManager.h"

using namespace irr;
using namespace core;

bool quit = false;

//!Same As Last Example
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
                quit = true;
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
    int32_t cameraDirUniformLocation;
    int32_t texUniformLocation[4];
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
    video::E_SHADER_CONSTANT_TYPE cameraDirUniformType;
    video::E_SHADER_CONSTANT_TYPE texUniformType[4];
public:
    SimpleCallBack() : cameraDirUniformLocation(-1), cameraDirUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::array<video::SConstantLocationNamePair>& constants)
    {
        for (size_t i=0; i<constants.size(); i++)
        {
            if (constants[i].name=="MVP")
            {
                mvpUniformLocation = constants[i].location;
                mvpUniformType = constants[i].type;
            }
            else if (constants[i].name=="cameraPos")
            {
                cameraDirUniformLocation = constants[i].location;
                cameraDirUniformType = constants[i].type;
            }
            else if (constants[i].name=="tex0")
            {
                texUniformLocation[0] = constants[i].location;
                texUniformType[0] = constants[i].type;
            }
            else if (constants[i].name=="tex3")
            {
                texUniformLocation[3] = constants[i].location;
                texUniformType[3] = constants[i].type;
            }
        }
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        core::vectorSIMDf modelSpaceCamPos;
        modelSpaceCamPos.set(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD_VIEW_INVERSE).getTranslation());
        if (cameraDirUniformLocation!=-1)
            services->setShaderConstant(&modelSpaceCamPos,cameraDirUniformLocation,cameraDirUniformType,1);
        if (mvpUniformLocation!=-1)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);

        int32_t id[] = {0,1,2,3};
        if (texUniformLocation[0]!=-1)
            services->setShaderTextures(id+0,texUniformLocation[0],texUniformType[0],1);
        if (texUniformLocation[3]!=-1)
            services->setShaderTextures(id+3,texUniformLocation[3],texUniformType[3],1);
    }

    virtual void OnUnsetMaterial() {}
};

class PostProcCallBack : public video::IShaderConstantSetCallBack
{
    int32_t sampleCountUniformLocation;
    video::E_SHADER_CONSTANT_TYPE sampleCountUniformType;
public:
    PostProcCallBack() {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::array<video::SConstantLocationNamePair>& constants)
    {
        /**
        Shader Unigorms get saved as Program (Shader state)
        So we can perma-assign texture slots to sampler uniforms
        **/
        int32_t id[] = {0,1,2,3};
        for (size_t i=0; i<constants.size(); i++)
        {
            if (constants[i].name=="tex0")
                services->setShaderTextures(id+0,constants[i].location,constants[i].type,1);
            else if (constants[i].name=="tex1")
                services->setShaderTextures(id+1,constants[i].location,constants[i].type,1);
            else if (constants[i].name=="sampleCount")
            {
                sampleCountUniformLocation = constants[i].location;
                sampleCountUniformType = constants[i].type;
            }
        }
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        if (sampleCountUniformLocation!=-1)
            services->setShaderConstant(&userData,sampleCountUniformLocation,sampleCountUniformType,1);
    }

    virtual void OnUnsetMaterial() {}
};


#include "irrpack.h"
struct ScreenQuadVertexStruct
{
    float Pos[3];
    uint8_t TexCoord[2];
} PACK_STRUCT;
#include "irrunpack.h"

int main()
{
	printf("Enter the number of samples to use for MSAA: ");

	uint32_t numberOfSamples = 8;
	std::cin >> numberOfSamples;
	// yeah yeah I'll add a query mechanism for the max in IVideoDriver
	if (numberOfSamples>64)
        numberOfSamples = 64;
    if (numberOfSamples<=1)
        numberOfSamples = 2;

	printf("\nPlease select the MSAA FBO attachment type to use:\n");
	printf(" (0 : default) Use Texture\n");
	printf(" (1) Use Renderbuffer\n");

	bool useRenderbuffer = false;
	uint32_t c;
	std::cin >> c;
	if (c==1)
        useRenderbuffer = true;

    //You may find while experimenting that you can only create a texture with 8 samples but renderbuffer with 32 !
    printf("\nUsing %s with %d samples.\n",useRenderbuffer ? "Renderbuffer":"Texture",numberOfSamples);

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

    SimpleCallBack* cb = new SimpleCallBack();
    video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        cb, //! Our Shader Callback
                                                        0); //! No custom user data
    cb->drop();



	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(250.0f);
    smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

        #define kInstanceSquareSize 10
	scene::ISceneNode* instancesToRemove[kInstanceSquareSize*kInstanceSquareSize] = {0};

	//! Test Loading of Obj
    scene::ICPUMesh* cpumesh = smgr->getMesh("../../media/dwarf.x");
    if (cpumesh&&cpumesh->getMeshType()==scene::EMT_ANIMATED_SKINNED)
    {
        scene::ISkinnedMeshSceneNode* anode = 0;
        scene::ICPUSkinnedMesh* animMesh = dynamic_cast<scene::ICPUSkinnedMesh*>(cpumesh);
        scene::IGPUMesh* gpumesh = driver->createGPUMeshFromCPU(cpumesh);
        smgr->getMeshCache()->removeMesh(cpumesh); //drops hierarchy

        for (size_t x=0; x<kInstanceSquareSize; x++)
        for (size_t z=0; z<kInstanceSquareSize; z++)
        {
            instancesToRemove[x+kInstanceSquareSize*z] = anode = smgr->addSkinnedMeshSceneNode(static_cast<scene::IGPUSkinnedMesh*>(gpumesh));
            anode->setScale(core::vector3df(0.05f));
            anode->setPosition(core::vector3df(x,0.f,z)*4.f);
            anode->setAnimationSpeed(18.f*float(x+1+(z+1)*kInstanceSquareSize)/float(kInstanceSquareSize*kInstanceSquareSize));
            anode->setMaterialType(newMaterialType);
            anode->setMaterialTexture(3,anode->getBonePoseTBO());
        }

        gpumesh->drop();
    }

    //! We use a renderbuffer because we don't intend on reading from it
    video::IRenderBuffer* colorRB=NULL,* depthRB=NULL;
    video::IMultisampleTexture* colorMT=NULL,* depthMT=NULL;
    scene::IGPUMeshBuffer* screenQuadMeshBuffer=NULL;
    video::SMaterial postProcMaterial;
    video::IFrameBuffer* framebuffer = driver->addFrameBuffer();
    if (useRenderbuffer)
    {
        colorRB = driver->addMultisampleRenderBuffer(numberOfSamples,params.WindowSize,video::ECF_A8R8G8B8);
        depthRB = driver->addMultisampleRenderBuffer(numberOfSamples,params.WindowSize,video::ECF_DEPTH32F);
        framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0,colorRB);
        framebuffer->attach(video::EFAP_DEPTH_ATTACHMENT,depthRB);
    }
    else
    {
        colorMT = driver->addMultisampleTexture(video::IMultisampleTexture::EMTT_2D,numberOfSamples,&params.WindowSize.Width,video::ECF_A8R8G8B8);
        depthMT = driver->addMultisampleTexture(video::IMultisampleTexture::EMTT_2D,numberOfSamples,&params.WindowSize.Width,video::ECF_DEPTH32F);
        framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0,colorMT);
        framebuffer->attach(video::EFAP_DEPTH_ATTACHMENT,depthMT);

        /**
        This extra stuff is to show off programmable resolve with a shader.
        **/
        screenQuadMeshBuffer = new scene::IGPUMeshBuffer();
        scene::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();
        screenQuadMeshBuffer->setMeshDataAndFormat(desc);
        desc->drop();

        ScreenQuadVertexStruct vertices[4];
        vertices[0].Pos[0] = -1.f;
        vertices[0].Pos[1] = -1.f;
        vertices[0].Pos[2] = 0.5f;
        vertices[0].TexCoord[0] = 0;
        vertices[0].TexCoord[1] = 0;
        vertices[1].Pos[0] = 1.f;
        vertices[1].Pos[1] = -1.f;
        vertices[1].Pos[2] = 0.5f;
        vertices[1].TexCoord[0] = 1;
        vertices[1].TexCoord[1] = 0;
        vertices[2].Pos[0] = -1.f;
        vertices[2].Pos[1] = 1.f;
        vertices[2].Pos[2] = 0.5f;
        vertices[2].TexCoord[0] = 0;
        vertices[2].TexCoord[1] = 1;
        vertices[3].Pos[0] = 1.f;
        vertices[3].Pos[1] = 1.f;
        vertices[3].Pos[2] = 0.5f;
        vertices[3].TexCoord[0] = 1;
        vertices[3].TexCoord[1] = 1;

        uint16_t indices_indexed16[] = {0,1,2,2,1,3};

        uint8_t* tmpMem = (uint8_t*)malloc(sizeof(vertices)+sizeof(indices_indexed16));
        memcpy(tmpMem,vertices,sizeof(vertices));
        memcpy(tmpMem+sizeof(vertices),indices_indexed16,sizeof(indices_indexed16));
        video::IGPUBuffer* buff = driver->createGPUBuffer(sizeof(vertices)+sizeof(indices_indexed16),tmpMem);
        free(tmpMem);

        desc->mapVertexAttrBuffer(buff,scene::EVAI_ATTR0,scene::ECPA_THREE,scene::ECT_FLOAT,sizeof(ScreenQuadVertexStruct),0);
        desc->mapVertexAttrBuffer(buff,scene::EVAI_ATTR1,scene::ECPA_TWO,scene::ECT_UNSIGNED_BYTE,sizeof(ScreenQuadVertexStruct),12); //this time we used unnormalized
        desc->mapIndexBuffer(buff);
        screenQuadMeshBuffer->setIndexBufferOffset(sizeof(vertices));
        screenQuadMeshBuffer->setIndexType(video::EIT_16BIT);
        screenQuadMeshBuffer->setIndexCount(6);
        buff->drop();

        PostProcCallBack* callBack = new PostProcCallBack();
        //! First need to make a material other than default to be able to draw with custom shader
        postProcMaterial.BackfaceCulling = false; //! Triangles will be visible from both sides
        postProcMaterial.ZBuffer = video::ECFN_ALWAYS; //! Ignore Depth Test
        postProcMaterial.ZWriteEnable = false; //! Why even write depth?
        postProcMaterial.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../screenquad.vert",
                                                                            "","","", //! No Geometry or Tessellation Shaders
                                                                            "../postproc.frag",
                                                                            3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only)
                                                                            callBack,
                                                                            NULL,0, //! Xform feedback stuff, irrelevant here
                                                                            numberOfSamples); //! custom user data
        //! Need to bind our Multisample Textures to the correct texture units upon draw
        postProcMaterial.setTexture(0,colorMT);
        postProcMaterial.setTexture(1,depthMT);
        callBack->drop();
    }


	uint64_t lastFPSTime = 0;

	while(device->run()&&(!quit))
	//if (device->isWindowActive())
	{
		driver->beginScene( false,false );

		driver->setRenderTarget(framebuffer);
		vectorSIMDf clearColor(1.f,1.f,1.f,1.f);
        driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,clearColor.pointer);
		driver->clearZBuffer();
        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer

        //yeah we dont have the state tracker yet
        glEnable(GL_MULTISAMPLE);
        smgr->drawAll();
        glDisable(GL_MULTISAMPLE);

        /**
        We could use the same codepath for MultisampleTextures as for Renderbuffers,
        since blit works on FBOs it would work here to as a resolve.

        But instead we show off programmable resolve with a shader.
        **/
        if (useRenderbuffer)
        {
            //notice how I dont even have to set the current FBO (render target) to 0 (the screen) for results to display
            const bool needToCopyDepth = false;
            driver->blitRenderTargets(framebuffer,0,needToCopyDepth);
        }
        else
        {
            driver->setRenderTarget(0);
            driver->setMaterial(postProcMaterial);
            driver->drawMeshBuffer(screenQuadMeshBuffer);
        }

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Builtin Nodes Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

	driver->removeFrameBuffer(framebuffer);
	if (useRenderbuffer)
    {
        driver->removeRenderBuffer(colorRB);
        driver->removeRenderBuffer(depthRB);
    }
    else
    {
        driver->removeMultisampleTexture(colorMT);
        driver->removeMultisampleTexture(depthMT);

        screenQuadMeshBuffer->drop();
    }

    for (size_t x=0; x<kInstanceSquareSize; x++)
    for (size_t z=0; z<kInstanceSquareSize; z++)
        instancesToRemove[x+kInstanceSquareSize*z]->remove();

    //create a screenshot
	video::IImage* screenshot = driver->createImage(video::ECF_A8R8G8B8,params.WindowSize);
        video::COpenGLExtensionHandler::extGlNamedFramebufferReadBuffer(0,GL_FRONT_LEFT);
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
	driver->writeImageToFile(screenshot,"./screenshot.png");
	screenshot->drop();

	device->drop();

	return 0;
}
