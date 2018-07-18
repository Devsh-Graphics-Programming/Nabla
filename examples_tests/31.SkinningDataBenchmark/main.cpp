#define _IRR_STATIC_LIB_
#include <iostream>
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"
#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/CSkinnedMesh.h"
#include "../source/Irrlicht/COpenGLDriver.h"
#include "../source/Irrlicht/COpenGLPersistentlyMappedBuffer.h"

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
		for (size_t i = 0; i<constants.size(); i++)
		{
			if (constants[i].name == "MVP")
			{
				mvpUniformLocation = constants[i].location;
				mvpUniformType = constants[i].type;
			}
			else if (constants[i].name == "cameraPos")
			{
				cameraDirUniformLocation = constants[i].location;
				cameraDirUniformType = constants[i].type;
			}
			else if (constants[i].name == "tex0")
			{
				texUniformLocation[0] = constants[i].location;
				texUniformType[0] = constants[i].type;
			}
			else if (constants[i].name == "tex3")
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
		if (cameraDirUniformLocation != -1)
			services->setShaderConstant(&modelSpaceCamPos, cameraDirUniformLocation, cameraDirUniformType, 1);
		if (mvpUniformLocation != -1)
			services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation, mvpUniformType, 1);

		int32_t id[] = { 0,1,2,3 };
		if (texUniformLocation[0] != -1)
			services->setShaderTextures(id + 0, texUniformLocation[0], texUniformType[0], 1);
		if (texUniformLocation[3] != -1)
			services->setShaderTextures(id + 3, texUniformLocation[3], texUniformType[3], 1);
	}

	virtual void OnUnsetMaterial() {}
};

size_t convertBuf1(const float* _src, float* _dst, const size_t _boneCnt, const size_t _instCnt)
{
    for (size_t i = 0u; i < _instCnt; ++i)
        memcpy(_dst + i*_boneCnt * 4 * 7, _src, sizeof(float) * _boneCnt * 4 * 7);
    printf("%d\n", sizeof(float) * _boneCnt * 4 * 7);
    return sizeof(float) * _boneCnt * 4 * 7 * _instCnt;
}
size_t convertBuf2(const float* _src, float* _dst, const size_t _boneCnt, const size_t _instCnt)
{
    float mat4x3[12], mat3[12];
    const size_t boneSize = (sizeof(mat4x3) + sizeof(mat3)) / sizeof(float);
    for (size_t i = 0u; i < _boneCnt; ++i)
    {
        memcpy(mat4x3, _src + 4*7*i, sizeof(mat4x3));
        for (size_t j = 0u; j < 3u; ++j)
            memcpy(mat3 + 4*j, _src + 4*7*i + sizeof(mat4x3)/sizeof(float) + 3*j, 3*sizeof(float));

        memcpy(_dst + boneSize*i, mat4x3, sizeof(mat4x3));
        memcpy(_dst + boneSize*i + sizeof(mat4x3)/sizeof(float), mat3, sizeof(mat3));
    }

    const size_t perInstance = boneSize * _boneCnt;

    for (size_t i = 1u; i < _instCnt; ++i)
        memcpy(_dst + i*perInstance, _dst, sizeof(float)*perInstance);

    return sizeof(float) * perInstance * _instCnt;
}
size_t convertBuf3(const float* _src, float* _dst, const size_t _boneCnt, const size_t _instCnt)
{
    const size_t size_mat4x3 = 12u; // 12 floats
    const size_t offset_mat3 = _boneCnt * size_mat4x3;
    for (size_t i = 0u; i < _boneCnt; ++i)
    {
        memcpy(_dst + i*size_mat4x3, _src + i*4*7, size_mat4x3*sizeof(float));
        // each mat3 column aligned to 4*sizeof(float)
        for (size_t j = 0u; j < 3u; ++j)
            memcpy(_dst + offset_mat3 + i*size_mat4x3 + j*4, _src + i*4*7 + size_mat4x3 + j*3, 3*sizeof(float));
    }

    const size_t perInstance = (3 + 3)*4 * _boneCnt;

    for (size_t i = 1u; i < _instCnt; ++i)
        memcpy(_dst + i*perInstance, _dst, sizeof(float)*perInstance);

    return sizeof(float) * perInstance * _instCnt;
}
size_t convertBuf4(const float* _src, float* _dst, const size_t _boneCnt, const size_t _instCnt)
{
    const size_t size_mat4x3 = 12u; // 12 floats
    const size_t 
        offset_mat4x3_0 = 0u,
        offset_mat4x3_1 = _boneCnt*4,
        offset_mat4x3_2 = _boneCnt*8,
        offset_mat3c0 = _boneCnt*12,
        offset_mat3c1 = _boneCnt*16,
        offset_mat3c2 = _boneCnt*20;
    for (size_t i = 0u; i < _boneCnt; ++i)
    {
        memcpy(_dst + offset_mat4x3_0 + i*4, _src + i*4*7 + 0*4, 4*sizeof(float));
        memcpy(_dst + offset_mat4x3_1 + i*4, _src + i*4*7 + 1*4, 4*sizeof(float));
        memcpy(_dst + offset_mat4x3_2 + i*4, _src + i*4*7 + 2*4, 4*sizeof(float));
        // each mat3 column aligned to 4*sizeof(float)
        memcpy(_dst + offset_mat3c0 + i*4, _src + i*4*7 + size_mat4x3 + 0*3, 3*sizeof(float));
        memcpy(_dst + offset_mat3c1 + i*4, _src + i*4*7 + size_mat4x3 + 1*3, 3*sizeof(float));
        memcpy(_dst + offset_mat3c2 + i*4, _src + i*4*7 + size_mat4x3 + 2*3, 3*sizeof(float));
    }

    const size_t perInstance = 4 * 6 * _boneCnt;

    for (size_t i = 1u; i < _instCnt; ++i)
        memcpy(_dst + i*perInstance, _dst, sizeof(float)*perInstance);

    return sizeof(float) * perInstance * _instCnt;
}
size_t convertBuf5(const float* _src, float* _dst, const size_t _boneCnt, const size_t _instCnt)
{
    size_t offsets[12+9];
    for (size_t i = 0u; i < sizeof(offsets)/sizeof(*offsets); ++i)
        offsets[i] = i*_boneCnt;

    for (size_t i = 0u; i < _boneCnt; ++i)
    {
        for (size_t o = 0u; o < sizeof(offsets)/sizeof(*offsets); ++o)
        {
            memcpy(_dst + offsets[o] + i, _src + i*4*7 + o, sizeof(float));
        }
    }

    const size_t perInstance = (12 + 9) * _boneCnt;

    for (size_t i = 1u; i < _instCnt; ++i)
        memcpy(_dst + i*perInstance, _dst, sizeof(float)*perInstance);

    return sizeof(float) * perInstance * _instCnt;
}

struct UBOManager
{
    UBOManager(size_t _sz, video::IVideoDriver* _drv) :
        ubo(_drv->createGPUBuffer(_sz, nullptr)),
        drv(_drv),
        mappedBuf(_drv->createPersistentlyMappedBuffer(4*_sz, nullptr, video::EGBA_WRITE, false, false)),
        updateNum{0u},
        fence{nullptr, nullptr, nullptr, nullptr}
    {}
    ~UBOManager()
    {
        mappedBuf->drop();
        ubo->drop();
    }

    void update(size_t _off, size_t _sz, const void* _data)
    {
        if (fence)
        {
            auto waitf = [this] {
                auto res = fence[updateNum]->waitCPU(10000000000ull);
                return (res == video::EDFR_CONDITION_SATISFIED || res == video::EDFR_ALREADY_SIGNALED);
            };
            while (!waitf())
            {
                fence[updateNum]->drop();
                fence[updateNum] = nullptr;
            }
        }

        memcpy(((uint8_t*)(dynamic_cast<video::COpenGLPersistentlyMappedBuffer*>(mappedBuf)->getPointer())) + updateNum*ubo->getSize() + _off, _data, _sz);
        video::COpenGLExtensionHandler::extGlFlushMappedNamedBufferRange(dynamic_cast<video::COpenGLPersistentlyMappedBuffer*>(mappedBuf)->getOpenGLName(), updateNum*ubo->getSize() + _off, _sz);

        drv->bufferCopy(mappedBuf, ubo, updateNum*ubo->getSize() + _off, _off, _sz);

        if (!fence)
            fence[updateNum] = drv->placeFence();

        updateNum = (updateNum + 1) % 4;
    }

    void bind(uint32_t _bnd, ptrdiff_t _off, ptrdiff_t _sz)
    {
        auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(drv)->getThreadContext());
        auto glbuf = static_cast<const video::COpenGLBuffer*>(ubo);
        auxCtx->setActiveUBO(_bnd, 1, &glbuf, &_off, &_sz);
    }

    video::IGPUBuffer* ubo;

private:
    video::IVideoDriver* drv;
    video::IGPUBuffer* mappedBuf;
    uint8_t updateNum;
    video::IDriverFence* fence[4];
};

//#define BENCH

int main(int _argCnt, char** _args)
{
#ifdef BENCH
    const dimension2d<uint32_t> WIN_SIZE(64u, 64u);
#else
    const dimension2d<uint32_t> WIN_SIZE(1280u, 720u);
#endif

	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = WIN_SIZE;
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();

    uint32_t method = 0u;
    if (_argCnt <= 1)
    {
        std::cout << "Method number [1;5]: ";
        std::cin >> method;
    }
    else
    {
        method = std::stoi(_args[1]);
    }
    --method;

	SimpleCallBack* cb = new SimpleCallBack();
	video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles((std::string("../vs") + std::to_string(method+1) + ".vert").c_str(),
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();

	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(nullptr, 100.0f, 0.01f);
	camera->setPosition(core::vector3df(-4, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(250.0f);
	smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

    UBOManager uboMgr(16*sizeof(float)/*mat4*/, driver);
    uboMgr.bind(0u, 0, uboMgr.ubo->getSize());
    // todo: fill ubo with MVP matrix (how to get this from engine?)

	scene::ICPUMesh* cpumesh = smgr->getMesh("dwarf.baw");

    using convfptr_t = size_t(*)(const float*, float*, const size_t, const size_t);
    convfptr_t convFunctions[5]{ &convertBuf1, &convertBuf2, &convertBuf3, &convertBuf4, &convertBuf5 };

#define INSTANCE_CNT 100
    auto anode = smgr->addSkinnedMeshSceneNode(static_cast<scene::IGPUSkinnedMesh*>(driver->createGPUMeshFromCPU(cpumesh)));
    // ^^ todo: this shouldn't be here. Draw mesh in some different way
    anode->setMaterialType(newMaterialType);
    anode->setScale(core::vector3df(0.5f));
    video::ITextureBufferObject* tbo = anode->getBonePoseTBO();
    video::IGPUBuffer* bonePosBuf = tbo->getBoundBuffer();
    auto bufcopy = driver->createGPUBuffer(bonePosBuf->getSize(), nullptr, true, true, video::EGBA_READ_WRITE);
    driver->bufferCopy(bonePosBuf, bufcopy, 0u, 0u, bonePosBuf->getSize());
    void* contents = video::COpenGLExtensionHandler::extGlMapNamedBuffer(static_cast<video::COpenGLBuffer*>(bufcopy)->getOpenGLName(), GL_READ_ONLY);
    void* newContents = malloc(5152/*max possible size of matrices for dwarf mesh*/ * INSTANCE_CNT);
    auto ssbuf = driver->createGPUBuffer(
        convFunctions[method]((float*)contents, (float*)newContents, anode->getBoneCount(), INSTANCE_CNT),
        newContents,
        false,
        false,
        video::EGBA_READ
    );
    free(newContents);
    video::COpenGLExtensionHandler::extGlUnmapNamedBuffer(static_cast<video::COpenGLBuffer*>(bufcopy)->getOpenGLName());

    if (method == 1)
    {
        tbo->bind(ssbuf, video::ITextureBufferObject::ETBOF_RGBA32F, 0, ssbuf->getSize());
        anode->setMaterialTexture(3, tbo);
    }
    else
    {
        auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(driver)->getThreadContext());
        const video::COpenGLBuffer* glbuf = static_cast<video::COpenGLBuffer*>(ssbuf);
        const ptrdiff_t off = 0, sz = glbuf->getSize();
        auxCtx->setActiveSSBO(0, 1, &glbuf, &off, &sz);
    }

    /*
    if (cpumesh)
    {
        for (size_t i = 0u; i < cpumesh->getMeshBufferCount(); ++i)
        {
            cpumesh->getMeshBuffer(i)->getMaterial().MaterialType = newMaterialType;
            cpumesh->getMeshBuffer(i)->setInstanceCount(INSTANCE_CNT);
        }
    }

    scene::IGPUMesh* gpumesh = driver->createGPUMeshFromCPU(cpumesh);
    */

#ifdef BENCH
    video::IFrameBuffer* fbo = driver->addFrameBuffer();
    fbo->attach(video::EFAP_DEPTH_ATTACHMENT, driver->addTexture(video::ITexture::ETT_2D, &WIN_SIZE.Width, 1, "Depth", video::ECF_DEPTH32F));
#endif

#define ITER_CNT 1000
    video::IQueryObject* queries[ITER_CNT];
    size_t itr = 0u;
	while (device->run() && itr < ITER_CNT && !quit)
	{
#ifdef BENCH
        driver->setRenderTarget(fbo, false);
#endif

		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        queries[itr] = driver->createElapsedTimeQuery();
        driver->beginQuery(queries[itr]);

		smgr->drawAll();
        /*
        if (gpumesh)
        {
            for (size_t i = 0u; i < gpumesh->getMeshBufferCount(); ++i)
                driver->drawMeshBuffer(gpumesh->getMeshBuffer(i));
        }
        */
        driver->endQuery(queries[itr]);

		driver->endScene();

#ifndef BENCH
		std::wostringstream str;
		str << L"Builtin Nodes Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

		device->setWindowCaption(str.str());
#endif

        ++itr;
	}

    size_t elapsed = 0u;
    for (size_t i = 0u; i < ITER_CNT; ++i)
    {
        uint32_t res;
        queries[i]->getQueryResult(&res);
        elapsed += res;
    }
    os::Printer::log("GPU time", std::to_string(elapsed));

	device->drop();

	return 0;
}
