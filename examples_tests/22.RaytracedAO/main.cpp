#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../ext/DebugDraw/CDraw3DLine.h"
#include "../../ext/ScreenShot/ScreenShot.h"

#include "../common/QToQuitEventReceiver.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "../../ext/MitsubaLoader/CMitsubaLoader.h"

using namespace irr;
using namespace core;

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
		video::E_MATERIAL_TYPE currentMat;

		int32_t mvpUniformLocation[video::EMT_COUNT+2];
		int32_t wUniformLocation[video::EMT_COUNT+2];
		int32_t colorUniformLocation[video::EMT_COUNT+2];
		int32_t nastyUniformLocation[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE mvpUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE wUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE colorUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE nastyUniformType[video::EMT_COUNT+2];
	public:
		SimpleCallBack() : currentMat(video::EMT_SOLID)
		{
			std::fill(mvpUniformLocation, reinterpret_cast<int32_t*>(mvpUniformType), -1);
		}

		virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
		{
			for (size_t i=0; i<constants.size(); i++)
			{
				if (constants[i].name == "MVP")
				{
					mvpUniformLocation[materialType] = constants[i].location;
					mvpUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "W")
				{
					wUniformLocation[materialType] = constants[i].location;
					wUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "color")
				{
					colorUniformLocation[materialType] = constants[i].location;
					colorUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "nasty")
				{
					nastyUniformLocation[materialType] = constants[i].location;
					nastyUniformType[materialType] = constants[i].type;
				}
			}
		}

		virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SGPUMaterial& material, const video::SGPUMaterial& lastMaterial)
		{
			currentMat = material.MaterialType;
			services->setShaderConstant(&material.AmbientColor, colorUniformLocation[currentMat], colorUniformType[currentMat], 1);
			services->setShaderConstant(&material.MaterialTypeParam, nastyUniformLocation[currentMat], nastyUniformType[currentMat], 1);
		}

		virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
		{
			if (wUniformLocation[currentMat]>=0)
				services->setShaderConstant(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD).pointer(), wUniformLocation[currentMat], wUniformType[currentMat], 1);
			if (mvpUniformLocation[currentMat]>=0)
				services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW).pointer(), mvpUniformLocation[currentMat], mvpUniformType[currentMat], 1);
		}

		virtual void OnUnsetMaterial() {}
};

int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 32;
	params.DriverType = video::EDT_OPENGL;
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	params.WindowSize = dimension2d<uint32_t>(1600, 900);
	IrrlichtDevice* device = createDeviceEx(params);
	if (device == 0)
		return 1; // could not create selected driver.

	//
	asset::SAssetBundle meshes;
	core::smart_refctd_ptr<ext::MitsubaLoader::CGlobalMitsubaMetadata> globalMeta;
	{
		io::IFileSystem* fs = device->getFileSystem();
		asset::IAssetManager* am = device->getAssetManager();

		am->addAssetLoader(core::make_smart_refctd_ptr<irr::ext::MitsubaLoader::CSerializedLoader>(am));
		am->addAssetLoader(core::make_smart_refctd_ptr<irr::ext::MitsubaLoader::CMitsubaLoader>(am));
#define QUICK_DEBUG
		std::string filePath;
		{
			//
			io::IFileArchive* arch = nullptr;
			device->getFileSystem()->addFileArchive("../../media/mitsuba/ditt.zip",io::EFAT_ZIP,"",&arch);
			if (!arch)
				return 2;

			filePath = "./newscene_inside.xml";
		}

		//! read cache results -- speeds up mesh generation
		{
			io::IReadFile* cacheFile = device->getFileSystem()->createAndOpenFile("../../tmp/normalCache101010.sse");
			if (cacheFile)
			{
				asset::normalCacheFor2_10_10_10Quant.resize(cacheFile->getSize() / sizeof(asset::QuantizationCacheEntry2_10_10_10));
				cacheFile->read(asset::normalCacheFor2_10_10_10Quant.data(), cacheFile->getSize());
				cacheFile->drop();

				//make sure its still ok
				std::sort(asset::normalCacheFor2_10_10_10Quant.begin(), asset::normalCacheFor2_10_10_10Quant.end());
			}
		}
		//! load the mitsuba scene
		meshes = am->getAsset(filePath, {});
		//! cache results -- speeds up mesh generation on second run
		{
			io::IWriteFile* cacheFile = device->getFileSystem()->createAndWriteFile("../../tmp/normalCache101010.sse");
			if (cacheFile)
			{
				cacheFile->write(asset::normalCacheFor2_10_10_10Quant.data(), asset::normalCacheFor2_10_10_10Quant.size() * sizeof(asset::QuantizationCacheEntry2_10_10_10));
				cacheFile->drop();
			}
		}
		
		auto contents = meshes.getContents();
		if (contents.first>=contents.second)
			return 2;

		auto firstmesh = *contents.first;
		if (!firstmesh)
			return 3;

		auto meta = firstmesh->getMetadata();
		if (!meta)
			return 4;
		assert(core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
		globalMeta = static_cast<ext::MitsubaLoader::IMeshMetadata*>(meta)->globalMetadata;
	}

/*
	// set resolution
	if (globalMeta->sensors.size())
	{
		const auto& film = globalMeta->sensors.front().film;
		params.WindowSize.Width = film.width;
		params.WindowSize.Height = film.height;
	}
*/

	scene::ISceneManager* smgr = device->getSceneManager();
	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	video::IVideoDriver* driver = device->getVideoDriver();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);

	SimpleCallBack* cb = new SimpleCallBack();
	video::E_MATERIAL_TYPE nonInstanced = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();

	core::vector<scene::ISceneNode*> nodes;
	{
		auto contents = meshes.getContents();
		contents.first += 133u;
		contents.second = contents.first+1u;
		auto gpumeshes = driver->getGPUObjectsFromAssets<asset::ICPUMesh>(contents.first, contents.second);
		auto cpuit = contents.first;
		for (auto gpuit = gpumeshes->begin(); gpuit!=gpumeshes->end(); gpuit++,cpuit++)
		{
			auto* meta = (*cpuit)->getMetadata();
			assert(meta && core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
			const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);
			const auto& instances = meshmeta->getInstances();

			const auto& gpumesh = *gpuit;
			for (auto i=0u; i<gpumesh->getMeshBufferCount(); i++)
				gpumesh->getMeshBuffer(i)->getMaterial().MaterialType = nonInstanced;

			//for (auto instance : instances)
			{
				auto node = smgr->addMeshSceneNode(core::smart_refctd_ptr(gpumesh));
				node->setRelativeTransformationMatrix(instances[0].getAsRetardedIrrlichtMatrix());
				nodes.push_back(node);
				node->grab();
				node->updateAbsolutePosition();
				node->setParent(nullptr);
			}
			//break;
		}
	}

	// camera and viewport
	bool leftHandedCamera = false;
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.1f);
	camera->setNearValue(20.f);
	camera->setFarValue(5000.f);
	camera->setLeftHanded(false);
	smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);

	uint64_t lastFPSTime = 0;
	float lastFastestMeshFrameNr = -1.f;
	{
		auto draw3DLine = ext::DebugDraw::CDraw3DLine::create(driver);
		while (device->run() && receiver.keepOpen())
		{
			driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

			//! This animates (moves) the camera and sets the transforms
			//! Also draws the meshbuffer
			smgr->drawAll();

			for (auto node : nodes)
			{
				core::vector<std::pair<ext::DebugDraw::S3DLineVertex, ext::DebugDraw::S3DLineVertex>> lines;
				draw3DLine->enqueueBox(lines,node->getBoundingBox(),1.f,0.f,0.f,1.f, node->getAbsoluteTransformation());
				draw3DLine->draw(lines);
				node->render();
			}

			driver->endScene();

			// display frames per second in window title
			uint64_t time = device->getTimer()->getRealTime();
			if (time - lastFPSTime > 1000)
			{
				std::wostringstream str;
				str << L"Mitsuba Loader Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

				device->setWindowCaption(str.str());
				lastFPSTime = time;
			}
		}
	}

	// create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	device->drop();
	return 0;
}
