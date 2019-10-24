#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"
#include "../../source/Irrlicht/COpenGLVAOSpec.h"

using namespace irr;
using namespace core;


class SimpleCallBack : public video::IShaderConstantSetCallBack
{
public:
	SimpleCallBack() {}

	virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
	{
	}

	virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
	{
		services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), 0u, video::ESCT_FLOAT_MAT4, 1);
	}

	virtual void OnUnsetMaterial() {}
};



struct DrawElementsIndirectCommand
{
	uint32_t count;
	uint32_t instanceCount;
	uint32_t firstIndex;
	uint32_t baseVertex;
	uint32_t baseInstance;
};

struct IndirectDrawInfo
{
	// maybe a unique transform in the future

	video::SGPUMaterial material;
	core::smart_refctd_ptr<asset::IMeshDataFormatDesc<video::IGPUBuffer> > pipeline;
	asset::E_PRIMITIVE_TYPE primitiveType;
	asset::E_INDEX_TYPE indexType;

	core::vector<DrawElementsIndirectCommand> indirectDrawData;
	core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawBuffer;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t MaxTexturesPerDraw = 2u; // need full bindless (by handle) to handle more than 32 textures
	core::vector<std::array<core::smart_refctd_ptr<video::IVirtualTexture>,MaxTexturesPerDraw> > textures;

	core::vector<video::IGPUMeshBuffer*> backup;
};


struct IndirectDrawKey : video::COpenGLVAOSpec::HashAttribs
{
	IndirectDrawKey(asset::IMeshDataFormatDesc<video::IGPUBuffer>* pipeline) : video::COpenGLVAOSpec::HashAttribs(dynamic_cast<video::COpenGLVAOSpec*>(pipeline)->getHash())
	{
		for (auto i = 0u; i < asset::EVAI_COUNT; i++)
		{
			auto attr = static_cast<asset::E_VERTEX_ATTRIBUTE_ID>(i);
			offset[i] = pipeline->getMappedBufferOffset(attr);
		}
	}

	inline bool operator<(const IndirectDrawKey& other) const
	{
		if (video::COpenGLVAOSpec::HashAttribs::operator<(other))
			return true;

		if (video::COpenGLVAOSpec::HashAttribs::operator!=(other))
			return false;

		// parents equal, now compare children
		for (auto i = 0u; i < asset::EVAI_COUNT; i++)
		{
			if (offset[i] < other.offset[i])
				return true;
			if (offset[i] > other.offset[i])
				return false;
		}
		return false;
	}

	inline bool operator!=(const IndirectDrawKey& other) const
	{
		if (video::COpenGLVAOSpec::HashAttribs::operator!=(other))
			return true;

		for (auto i = 0u; i < asset::EVAI_COUNT; i++)
		{
			if (offset[i] != other.offset[i])
				return true;
		}
		return false;
	}

	inline bool operator==(const IndirectDrawKey& other) const
	{
		return !((*this) != other);
	}

	uint32_t offset[asset::EVAI_COUNT];
};

int main()
{
	srand(time(0));
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	device->getCursorControl()->setVisible(false);

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);


	video::IVideoDriver* driver = device->getVideoDriver();

	SimpleCallBack* cb = new SimpleCallBack();
	video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();



	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0, 100.0f, 1.f);
	camera->setPosition(core::vector3df(-4, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(1.f);
	camera->setFarValue(10000.0f);
	smgr->setActiveCamera(camera);

	io::IFileSystem* fs = device->getFileSystem();
	auto am = device->getAssetManager();

	//! Load big-ass sponza model
	// really want to get it working with a "../../media/sponza.zip?sponza.obj" path handling
	fs->addFileArchive("../../media/sponza.zip");



	using pipeline_type = asset::IMeshDataFormatDesc<video::IGPUBuffer>;
	core::map<IndirectDrawKey, IndirectDrawInfo> indirectDraws;
	//{
		//! read cache results -- speeds up mesh generation
		{
			io::IReadFile* cacheFile = device->getFileSystem()->createAndOpenFile("./normalCache101010.sse");
			if (cacheFile)
			{
				asset::normalCacheFor2_10_10_10Quant.resize(cacheFile->getSize() / sizeof(asset::QuantizationCacheEntry2_10_10_10));
				cacheFile->read(asset::normalCacheFor2_10_10_10Quant.data(), cacheFile->getSize());
				cacheFile->drop();

				//make sure its still ok
				std::sort(asset::normalCacheFor2_10_10_10Quant.begin(), asset::normalCacheFor2_10_10_10Quant.end());
			}
		}

		asset::IAssetLoader::SAssetLoadParams lparams; // crashes OBJ loader with those (0u, nullptr, asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES);
		auto asset = am->getAsset("sponza.obj", lparams);
		auto cpumesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*asset.getContents().first);
		auto gpumesh = driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get()) + 1)->operator[](0);

		//! cache results -- speeds up mesh generation on second run
		{
			io::IWriteFile* cacheFile = device->getFileSystem()->createAndWriteFile("./normalCache101010.sse");
			cacheFile->write(asset::normalCacheFor2_10_10_10Quant.data(), asset::normalCacheFor2_10_10_10Quant.size() * sizeof(asset::QuantizationCacheEntry2_10_10_10));
			cacheFile->drop();
		}


		for (auto i = 0u; i < gpumesh->getMeshBufferCount(); i++)
		{
			auto* mb = gpumesh->getMeshBuffer(i);
			pipeline_type* pipeline = mb->getMeshDataAndFormat();
			const auto& descriptors = mb->getMaterial();

			auto& info = indirectDraws[IndirectDrawKey(pipeline)];
			if (info.indirectDrawData.size() == 0u)
			{
				info.material = mb->getMaterial();
				info.material.MaterialType = newMaterialType;
				info.pipeline = core::smart_refctd_ptr<asset::IMeshDataFormatDesc<video::IGPUBuffer> >(pipeline);
				info.primitiveType = mb->getPrimitiveType();
				info.indexType = mb->getIndexType();
				assert(info.indexType != asset::EIT_UNKNOWN);
			}
			else
			{
				assert(info.primitiveType == mb->getPrimitiveType());
				assert(info.indexType == mb->getIndexType());

				assert(info.backup.front()->getMeshDataAndFormat()->getIndexBuffer() == pipeline->getIndexBuffer());
				for (auto j = 0u; j < asset::EVAI_COUNT; j++)
				{
					auto attr = static_cast<asset::E_VERTEX_ATTRIBUTE_ID>(j);
					assert(info.backup.front()->getMeshDataAndFormat()->getMappedBuffer(attr) == pipeline->getMappedBuffer(attr));
					assert(info.backup.front()->getMeshDataAndFormat()->getMappedBufferOffset(attr) == pipeline->getMappedBufferOffset(attr));
					assert(info.backup.front()->getMeshDataAndFormat()->getMappedBufferStride(attr) == pipeline->getMappedBufferStride(attr));
				}
			}

			DrawElementsIndirectCommand cmd;
			cmd.count = mb->getIndexCount();
			cmd.instanceCount = mb->getInstanceCount();
			cmd.firstIndex = mb->getIndexBufferOffset()/(info.indexType!=asset::EIT_32BIT ? sizeof(uint16_t):sizeof(uint32_t));
			cmd.baseVertex = mb->getBaseVertex();
			cmd.baseInstance = mb->getBaseInstance();
			info.indirectDrawData.push_back(std::move(cmd));

			decltype(IndirectDrawInfo::textures)::value_type textures;
			for (uint32_t i=0u; i<IndirectDrawInfo::MaxTexturesPerDraw; i++)
				textures[i] = mb->getMaterial().TextureLayer[i].Texture;
			info.textures.push_back(std::move(textures));

			info.backup.emplace_back(mb);
		}
	//}
	for (auto& draw : indirectDraws)
	{
		draw.second.indirectDrawBuffer = core::smart_refctd_ptr<video::IGPUBuffer>(driver->createFilledDeviceLocalGPUBufferOnDedMem(draw.second.indirectDrawData.size()*sizeof(DrawElementsIndirectCommand),draw.second.indirectDrawData.data()));
	}


	uint64_t lastFPSTime = 0;

	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

		//! Draw the view
		smgr->drawAll();

		for (auto& draw : indirectDraws)
		{
			auto& info = draw.second;
			if (!info.indirectDrawBuffer)
				continue; 

			driver->setMaterial(info.material);
			{
				GLuint textures[1024u] = { 0u };
				GLenum targets[1024u] = { GL_INVALID_ENUM };

				uint32_t count = 0u;
				for (const auto& usedTexs : info.textures)
				for (auto& tex : usedTexs)
				{
					auto* gltex = dynamic_cast<video::COpenGLTexture*>(tex.get());
					if (gltex)
					{
						textures[count] = gltex->getOpenGLName();
						targets[count] = gltex->getOpenGLTextureType();
					}
					else
					{
						textures[count] = 0u;
						targets[count] = GL_INVALID_ENUM;
					}
					count++;
				}

				video::COpenGLExtensionHandler::extGlBindTextures(0u, count, textures, targets);
			}
			driver->drawIndexedIndirect(info.pipeline.get(), info.primitiveType, info.indexType, info.indirectDrawBuffer.get(), 0, info.indirectDrawBuffer->getSize()/sizeof(DrawElementsIndirectCommand), sizeof(DrawElementsIndirectCommand));
		}

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream sstr;
			sstr << L"Builtin Nodes Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(sstr.str().c_str());
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
