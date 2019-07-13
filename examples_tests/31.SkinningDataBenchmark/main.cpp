#define _IRR_STATIC_LIB_
#include <iostream>
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"
#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"
#include "../source/Irrlicht/COpenGLTextureBufferObject.h"

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

size_t convertBuf1(const float* _src, float* _dst, const size_t _boneCnt, const size_t _instCnt)
{
    for (size_t i = 0u; i < _boneCnt; ++i)
        memcpy(_dst + i*4*6, _src + i*4*7, sizeof(float) * _boneCnt * 4 * 6);

    const size_t perInstance = sizeof(float)*_boneCnt*4*6;

    for (size_t i = 1u; i < _instCnt; ++i)
        memcpy(_dst + i*perInstance, _dst, sizeof(float)*perInstance);

    return sizeof(float) * perInstance * _instCnt;
}
size_t convertBuf2(const float* _src, float* _dst, const size_t _boneCnt, const size_t _instCnt)
{
    // by mat3x4 (3 rows, 4 columns, column-major layout) i mean glsl's mat4x3...
    float mat3x4[16]; // layout: m00|m10|m20|---|m01|m11|m21|---|m02|m12|m22|---|m03|m13|m23|--- ("---" is unused padding)
    float mat3[12];
    const size_t size_mat3x4_src = 12u;

    const size_t boneSize = (sizeof(mat3x4) + sizeof(mat3)) / sizeof(float);
    for (size_t i = 0u; i < _boneCnt; ++i)
    {
        for (size_t j = 0u; j < 4u; ++j)
            memcpy(mat3x4 + 4*j, _src + 4*7*i + 3*j, 3*sizeof(float));
        for (size_t j = 0u; j < 3u; ++j)
            memcpy(mat3 + 4*j, _src + 4*7*i + size_mat3x4_src + 3*j, 3*sizeof(float));

        memcpy(_dst + boneSize*i, mat3x4, sizeof(mat3x4));
        memcpy(_dst + boneSize*i + sizeof(mat3x4)/sizeof(float), mat3, sizeof(mat3));
    }

    const size_t perInstance = boneSize * _boneCnt;

    for (size_t i = 1u; i < _instCnt; ++i)
        memcpy(_dst + i*perInstance, _dst, sizeof(float)*perInstance);

    return sizeof(float) * perInstance * _instCnt;
}
size_t convertBuf3(const float* _src, float* _dst, const size_t _boneCnt, const size_t _instCnt)
{
    const size_t size_mat3_dst = 12u; // 12 floats
    const size_t size_mat4x3_src = 12u; // 12 floats
    const size_t size_mat4x3_dst = 16u; // 16 floats
    const size_t offset_mat3_dst = _boneCnt * size_mat4x3_dst;
    for (size_t i = 0u; i < _boneCnt; ++i)
    {
        //memcpy(_dst + i*size_mat4x3_dst, _src + i*4*7, size_mat4x3*sizeof(float));
        for (size_t j = 0u; j < 4u; ++j)
            memcpy(_dst + i*size_mat4x3_dst + 4*j, _src + i*4*7 + 3*j, 3*sizeof(float));
        // each mat3 column aligned to 4*sizeof(float)
        for (size_t j = 0u; j < 3u; ++j)
            memcpy(_dst + offset_mat3_dst + i*size_mat3_dst + j*4, _src + i*4*7 + size_mat4x3_src + j*3, 3*sizeof(float));
    }

    const size_t perInstance = (size_mat3_dst + size_mat4x3_dst) * _boneCnt;

    for (size_t i = 1u; i < _instCnt; ++i)
        memcpy(_dst + i*perInstance, _dst, sizeof(float)*perInstance);

    return sizeof(float) * perInstance * _instCnt;
}
size_t convertBuf4(const float* _src, float* _dst, const size_t _boneCnt, const size_t _instCnt)
{
    const size_t size_mat4x3_src = 12u; // 12 floats
    const size_t 
        offset_mat4x3_0 = 0u,
        offset_mat4x3_1 = _boneCnt*4,
        offset_mat4x3_2 = _boneCnt*8,
        offset_mat4x3_3 = _boneCnt*12,
        offset_mat3_0 = _boneCnt*16,
        offset_mat3_1 = _boneCnt*20,
        offset_mat3_2 = _boneCnt*24;
    for (size_t i = 0u; i < _boneCnt; ++i)
    {
        memcpy(_dst + offset_mat4x3_0 + i*4, _src + i*4*7 + 0*3, 3*sizeof(float));
        memcpy(_dst + offset_mat4x3_1 + i*4, _src + i*4*7 + 1*3, 3*sizeof(float));
        memcpy(_dst + offset_mat4x3_2 + i*4, _src + i*4*7 + 2*3, 3*sizeof(float));
        memcpy(_dst + offset_mat4x3_3 + i*4, _src + i*4*7 + 3*3, 3*sizeof(float));
        // each mat3 column aligned to 4*sizeof(float)
        memcpy(_dst + offset_mat3_0 + i*4, _src + i*4*7 + size_mat4x3_src + 0*3, 3*sizeof(float));
        memcpy(_dst + offset_mat3_1 + i*4, _src + i*4*7 + size_mat4x3_src + 1*3, 3*sizeof(float));
        memcpy(_dst + offset_mat3_2 + i*4, _src + i*4*7 + size_mat4x3_src + 2*3, 3*sizeof(float));
    }

    const size_t perInstance = 4 * 7 * _boneCnt;

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
        drv(_drv),
        updateNum{0u},
        fence{nullptr, nullptr, nullptr, nullptr}
    {
        video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
        reqs.vulkanReqs.alignment = 4;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
        reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE | video::IDriverMemoryAllocation::EMCF_COHERENT;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;
        reqs.vulkanReqs.size = 4*_sz;
        mappedBuf = drv->createGPUBufferOnDedMem(reqs);

        reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CANNOT_MAP;
        reqs.vulkanReqs.size = _sz;
        ubo = drv->createGPUBufferOnDedMem(reqs);

        mappedMem = (uint8_t*)mappedBuf->getBoundMemory()->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_WRITE, { updateNum*ubo->getSize(), ubo->getSize() });
    }
    ~UBOManager()
    {
        mappedBuf->getBoundMemory()->unmapMemory();
        mappedBuf->drop();
        ubo->drop();
    }

    void update(size_t _off, size_t _sz, const void* _data)
    {
        if (fence[updateNum])
        {
            auto waitf = [this] {
                auto res = fence[updateNum]->waitCPU(10000000000ull);
                return (res == video::EDFR_CONDITION_SATISFIED || res == video::EDFR_ALREADY_SIGNALED);
            };
			while (!waitf()) {}
			fence[updateNum] = nullptr;
        }

        memcpy(mappedMem + updateNum*ubo->getSize() + _off, _data, _sz);
        video::COpenGLExtensionHandler::extGlFlushMappedNamedBufferRange(dynamic_cast<video::COpenGLBuffer*>(mappedBuf)->getOpenGLName(), updateNum*ubo->getSize() + _off, _sz);

        drv->copyBuffer(mappedBuf, ubo, updateNum*ubo->getSize() + _off, _off, _sz);
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
    uint8_t* mappedMem;
    uint8_t updateNum;
    core::smart_refctd_ptr<video::IDriverFence> fence[4];
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

	video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles((std::string("../vs") + std::to_string(method+1) + ".vert").c_str(),
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		nullptr,
		0); //! No custom user data

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

    asset::IAssetManager& assetMgr = device->getAssetManager();
    asset::IAssetLoader::SAssetLoadParams lparams;
    asset::ICPUMesh* cpumesh = static_cast<asset::ICPUMesh*>(assetMgr.getAsset("../../media/dwarf.baw", lparams));

    using convfptr_t = size_t(*)(const float*, float*, const size_t, const size_t);
    convfptr_t convFunctions[5]{ &convertBuf1, &convertBuf2, &convertBuf3, &convertBuf4, &convertBuf5 };

    video::SGPUMaterial smaterial;
    smaterial.MaterialType = newMaterialType;

#define INSTANCE_CNT 100
    video::IGPUMesh* gpumesh = driver->getGPUObjectsFromAssets(&cpumesh, (&cpumesh)+1).front();
    for (size_t i = 0u; i < gpumesh->getMeshBufferCount(); ++i)
        gpumesh->getMeshBuffer(i)->setInstanceCount(INSTANCE_CNT);

    io::IReadFile* ifile = device->getFileSystem()->createAndOpenFile("../tbo_dump.rgba32f");
    void* contents = malloc(ifile->getSize());
    void* newContents = malloc(51520/*some reasonable value surely bigger than any of 5 bone transforms arrangements in the buf*/ * INSTANCE_CNT);
    ifile->read(contents, ifile->getSize());

    const size_t actualSSBODataSize = convFunctions[method]((float*)contents, (float*)newContents, 46, INSTANCE_CNT);
    video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
    reqs.vulkanReqs.alignment = 4;
    reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
    reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DONT_KNOW;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ | video::IDriverMemoryAllocation::EMCF_COHERENT;
    reqs.prefersDedicatedAllocation = true;
    reqs.requiresDedicatedAllocation = true;
    reqs.vulkanReqs.size = alignUp(actualSSBODataSize, 16u);
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CANNOT_MAP;
    auto ssbuf = driver->createGPUBufferOnDedMem(reqs, true);
    ssbuf->updateSubRange({ 0, ssbuf->getSize() }, newContents);
    free(newContents);
    free(contents);
    ifile->drop();

    video::ITextureBufferObject* newTbo = method==0 ?
        new video::COpenGLTextureBufferObject(static_cast<video::COpenGLBuffer*>(ssbuf), video::ITextureBufferObject::ETBOF_RGBA32F, 0, ssbuf->getSize()) :
        nullptr;
    if (method == 0)
    {
        assert(newTbo->getBoundBuffer() != nullptr);
        
        smaterial.setTexture(3u, newTbo);
    }
    else
    {
        auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(driver)->getThreadContext());
        const video::COpenGLBuffer* glbuf = static_cast<video::COpenGLBuffer*>(ssbuf);
        const ptrdiff_t off = 0, sz = glbuf->getSize();
        auxCtx->setActiveSSBO(0, 1, &glbuf, &off, &sz);
    }

#ifdef BENCH
    video::IFrameBuffer* fbo = driver->addFrameBuffer();
    fbo->attach(video::EFAP_DEPTH_ATTACHMENT, driver->createGPUTexture(video::ITexture::ETT_2D, &WIN_SIZE.Width, 1, asset::EF_D32_SFLOAT));
#endif

#define ITER_CNT 1000
    video::IQueryObject* queries[ITER_CNT];
    size_t itr = 0u;
	while (device->run() && itr < ITER_CNT && !quit)
	{
#ifdef BENCH
        driver->setRenderTarget(fbo, false);
#endif

        uboMgr.update(0u, 16*sizeof(float), driver->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer());
        uboMgr.bind(0u, 0, 16*sizeof(float));

		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        queries[itr] = driver->createElapsedTimeQuery();
        driver->beginQuery(queries[itr]);

		smgr->drawAll();
        if (gpumesh)
        {
            for (size_t i = 0u; i < gpumesh->getMeshBufferCount(); ++i)
            {
                smaterial = gpumesh->getMeshBuffer(i)->getMaterial();
                smaterial.MaterialType = newMaterialType;
                if (method == 0)
                    smaterial.setTexture(3u, newTbo);

                driver->setMaterial(smaterial);
                driver->drawMeshBuffer(gpumesh->getMeshBuffer(i));
            }
        }
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
    for (size_t i = 0u; i < itr; ++i)
    {
        uint32_t res;
        queries[i]->getQueryResult(&res);
        elapsed += res;
    }
    std::stringstream ss;
    ss << "GPU time (";
    ss << itr;
    ss << " frames)";
    os::Printer::log(ss.str(), std::to_string(elapsed));

    if (newTbo)
        newTbo->drop();
	device->drop();

	return 0;
}
