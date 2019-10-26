#define _IRR_STATIC_LIB_
#include <irrlicht.h>

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
		int32_t colorUniformLocation[video::EMT_COUNT+2];
		int32_t nastyUniformLocation[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE mvpUniformType[video::EMT_COUNT+2];
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
			if (mvpUniformLocation[currentMat]>=0)
				services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation[currentMat], mvpUniformType[currentMat], 1);
		}

		virtual void OnUnsetMaterial() {}
};


class CullBack : public video::IShaderConstantSetCallBack
{
	inline static const char* uniformNames[] =
	{
		"ProjViewWorldMat",
		"LoDInvariantMinEdge",
		"LoDInvariantMaxEdge"
	};

	enum E_UNIFORM
	{
		EU_PROJ_VIEW_WORLD_MAT = 0,
		EU_LOD_INVARIANT_MIN_EDGE,
		EU_LOD_INVARIANT_MAX_EDGE,
		EU_COUNT
	};

    int32_t uniformLocation[EU_COUNT];
    video::E_SHADER_CONSTANT_TYPE uniformType[EU_COUNT];
public:
    core::aabbox3df instanceLoDInvariantBBox;

    CullBack()
    {
        for (size_t i=0; i<EU_COUNT; i++)
            uniformLocation[i] = -1;
    }

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        for (size_t i=0; i<constants.size(); i++)
        for (size_t j=0; j<EU_COUNT; j++)
        {
            if (constants[i].name==uniformNames[j])
            {
                uniformLocation[j] = constants[i].location;
                uniformType[j] = constants[i].type;
                break;
            }
        }
    }

    virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SGPUMaterial& material, const video::SGPUMaterial& lastMaterial) {}

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        if (uniformLocation[EU_PROJ_VIEW_WORLD_MAT]>=0)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),uniformLocation[EU_PROJ_VIEW_WORLD_MAT],uniformType[EU_PROJ_VIEW_WORLD_MAT],1);

        if (uniformLocation[EU_LOD_INVARIANT_MIN_EDGE]>=0)
        {
            services->setShaderConstant(&instanceLoDInvariantBBox.MinEdge,uniformLocation[EU_LOD_INVARIANT_MIN_EDGE],uniformType[EU_LOD_INVARIANT_MIN_EDGE],1);
            services->setShaderConstant(&instanceLoDInvariantBBox.MaxEdge,uniformLocation[EU_LOD_INVARIANT_MAX_EDGE],uniformType[EU_LOD_INVARIANT_MAX_EDGE],1);
        }
    }

    virtual void OnUnsetMaterial() {}
};

core::smart_refctd_ptr<asset::IMeshDataFormatDesc<video::IGPUBuffer> > vaoSetupOverride(scene::ISceneManager* smgr, video::IGPUBuffer* instanceDataBuffer, const size_t& dataSizePerInstanceOutput, const asset::IMeshDataFormatDesc<video::IGPUBuffer>* oldVAO, void* userData)
{
	video::IVideoDriver* driver = smgr->getVideoDriver();
	auto vao = driver->createGPUMeshDataFormatDesc();

	//
	for (size_t k=0; k<asset::EVAI_COUNT; k++)
	{
		asset::E_VERTEX_ATTRIBUTE_ID attrId = (asset::E_VERTEX_ATTRIBUTE_ID)k;
		if (!oldVAO->getMappedBuffer(attrId))
			continue;

		vao->setVertexAttrBuffer(	core::smart_refctd_ptr<video::IGPUBuffer>(const_cast<video::IGPUBuffer*>(oldVAO->getMappedBuffer(attrId))),
									attrId,oldVAO->getAttribFormat(attrId), oldVAO->getMappedBufferStride(attrId),oldVAO->getMappedBufferOffset(attrId),
									oldVAO->getAttribDivisor(attrId));
	}

	// I know what attributes are unused in my mesh and I've set up the shader to use thse as instance data
	constexpr auto stride = 25*sizeof(float);
	vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR4,asset::EF_R32G32B32A32_SFLOAT,stride,0,1);
	vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR5,asset::EF_R32G32B32A32_SFLOAT,stride,4*sizeof(float),1);
	vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR6,asset::EF_R32G32B32A32_SFLOAT,stride,8*sizeof(float),1);
	vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR7,asset::EF_R32G32B32A32_SFLOAT,stride,12*sizeof(float),1);
	vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR8,asset::EF_R32G32B32_SFLOAT,stride,16*sizeof(float),1);
	vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR9,asset::EF_R32G32B32_SFLOAT,stride,19*sizeof(float),1);
	vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR10,asset::EF_R32G32B32_SFLOAT,stride,22*sizeof(float),1);


	if (oldVAO->getIndexBuffer())
		vao->setIndexBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(const_cast<video::IGPUBuffer*>(oldVAO->getIndexBuffer())));

	return vao;
}

int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_NULL;
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon

	//
	asset::SAssetBundle meshes;
	core::smart_refctd_ptr<ext::MitsubaLoader::CGlobalMitsubaMetadata> globalMeta;
	{
		IrrlichtDevice* device = createDeviceEx(params);


		io::IFileSystem* fs = device->getFileSystem();
		asset::IAssetManager* am = device->getAssetManager();

		am->addAssetLoader(core::make_smart_refctd_ptr<irr::ext::MitsubaLoader::CSerializedLoader>(am));
		am->addAssetLoader(core::make_smart_refctd_ptr<irr::ext::MitsubaLoader::CMitsubaLoader>(am));

		std::string filePath = "../../media/mitsuba/staircase2.zip";
	//#define MITSUBA_LOADER_TESTS
	#ifndef MITSUBA_LOADER_TESTS
		pfd::message("Choose file to load", "Choose mitsuba XML file to load or ZIP containing an XML. \nIf you cancel or choosen file fails to load staircase will be loaded.", pfd::choice::ok);
		pfd::open_file file("Choose XML or ZIP file", "../../media/mitsuba", { "ZIP files (.zip)", "*.zip", "XML files (.xml)", "*.xml"});
		if (!file.result().empty())
			filePath = file.result()[0];
	#endif
		if (core::hasFileExtension(io::path(filePath.c_str()), "zip", "ZIP"))
		{
			io::IFileArchive* arch = nullptr;
			device->getFileSystem()->addFileArchive(filePath.c_str(),io::EFAT_ZIP,"",&arch);
			if (!arch)
				device->getFileSystem()->addFileArchive("../../media/mitsuba/staircase2.zip", io::EFAT_ZIP, "", &arch);
			if (!arch)
				return 2;

			auto flist = arch->getFileList();
			if (!flist)
				return 3;
			auto files = flist->getFiles();

			for (auto it=files.begin(); it!=files.end(); )
			{
				if (core::hasFileExtension(it->FullName, "xml", "XML"))
					it++;
				else
					it = files.erase(it);
			}
			if (files.size() == 0u)
				return 4;

			std::cout << "Choose File (0-" << files.size() - 1ull << "):" << std::endl;
			for (auto i = 0u; i < files.size(); i++)
				std::cout << i << ": " << files[i].FullName.c_str() << std::endl;
			uint32_t chosen = 0;
	#ifndef MITSUBA_LOADER_TESTS
			std::cin >> chosen;
	#endif
			if (chosen >= files.size())
				chosen = 0u;

			filePath = files[chosen].FullName.c_str();
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
			cacheFile->write(asset::normalCacheFor2_10_10_10Quant.data(), asset::normalCacheFor2_10_10_10Quant.size() * sizeof(asset::QuantizationCacheEntry2_10_10_10));
			cacheFile->drop();
		}

		device->drop();

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


	// recreate wth resolution
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	// set resolution
	if (globalMeta->sensors.size())
	{
		const auto& film = globalMeta->sensors.front().film;
		params.WindowSize.Width = film.width;
		params.WindowSize.Height = film.height;
	}
	params.DriverType = video::EDT_OPENGL;
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


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
	video::E_MATERIAL_TYPE instanced = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh_instanced.vert",
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();

	//instancing juice
	video::SGPUMaterial cullingXFormFeedbackShader;
	CullBack* cullback = new CullBack();
	{
		const char* xformFeedbackOutputs[] =
		{
			"worldViewProjMatCol0",
			"worldViewProjMatCol1",
			"worldViewProjMatCol2",
			"worldViewProjMatCol3",
			"tposeInverseWorldMatCol0",
			"tposeInverseWorldMatCol1",
			"tposeInverseWorldMatCol2"
		};
		cullingXFormFeedbackShader.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../culling.vert", "", "", "../culling.geom", "", 3, video::EMT_SOLID, cullback, xformFeedbackOutputs, 7u);
		cullback->drop();
		cullingXFormFeedbackShader.RasterizerDiscard = true; 
	}

	core::aabbox3df sceneBound;
	{
		auto gpumeshes = driver->getGPUObjectsFromAssets<asset::ICPUMesh>(meshes.getContents().first, meshes.getContents().second);
		auto cpuit = meshes.getContents().first;
		for (auto gpuit = gpumeshes->begin(); gpuit!=gpumeshes->end(); gpuit++,cpuit++)
		{
			auto* meta = (*cpuit)->getMetadata();
			assert(meta && core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
			const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);
			const auto& instances = meshmeta->getInstances();

			const auto& gpumesh = *gpuit;
#define INSTANCING_THRESHOLD 256u // TODO: when shader pipeline API comes, you can reduce this back to 1 as we need a nicer single-pass GPU-driven rendering system
			for (auto i=0u; i<gpumesh->getMeshBufferCount(); i++)
				gpumesh->getMeshBuffer(i)->getMaterial().MaterialType = instances.size()<INSTANCING_THRESHOLD ? nonInstanced:instanced;

			if (instances.size()<INSTANCING_THRESHOLD)
			for (auto instance : instances)
			{
				auto node = smgr->addMeshSceneNode(core::smart_refctd_ptr(gpumesh));
				node->setRelativeTransformationMatrix(instance.getAsRetardedIrrlichtMatrix());
				node->updateAbsolutePosition();
				sceneBound.addInternalBox(node->getTransformedBoundingBox());
			}
			else
			{
				auto node = smgr->addMeshSceneNodeInstanced();
				node->setBBoxUpdateEnabled();
				{
					core::vector<scene::IMeshSceneNodeInstanced::MeshLoD> LevelsOfDetail(1);
					LevelsOfDetail[0].mesh = gpumesh.get();
					LevelsOfDetail[0].lodDistance = FLT_MAX;

					bool success = node->setLoDMeshes(LevelsOfDetail, 25*sizeof(float), cullingXFormFeedbackShader, vaoSetupOverride);
					assert(success);
					cullback->instanceLoDInvariantBBox = node->getLoDInvariantBBox();
				}
				for (auto instance : instances)
					node->addInstance(instance.getAsRetardedIrrlichtMatrix());
				node->updateAbsolutePosition();
				sceneBound.addInternalBox(node->getTransformedBoundingBox());
				node->setAutomaticCulling(scene::EAC_FRUSTUM_BOX);
			}
		}
	}

	// camera and viewport
	bool leftHandedCamera = false;
	scene::ICameraSceneNode* camera = nullptr;
	core::recti viewport(core::position2di(0,0), core::position2di(params.WindowSize.Width,params.WindowSize.Height));

	auto isOkSensorType = [](const ext::MitsubaLoader::CElementSensor& sensor) -> bool {
		return sensor.type==ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE || sensor.type==ext::MitsubaLoader::CElementSensor::Type::THINLENS;
	};
	if (globalMeta->sensors.size() && isOkSensorType(globalMeta->sensors.front()))
	{
		const auto& sensor = globalMeta->sensors.front();
		const auto& film = sensor.film;
		viewport = core::recti(core::position2di(film.cropOffsetX,film.cropOffsetY), core::position2di(film.cropWidth,film.cropHeight));

		auto extent = sceneBound.getExtent();
		camera = smgr->addCameraSceneNodeFPS(nullptr,100.f,core::min(extent.X,extent.Y,extent.Z)*0.0002f);
		// need to extract individual components
		{
			auto relativeTransform = sensor.transform.matrix.extractSub3x4();
			auto pos = relativeTransform.getTranslation();
			camera->setPosition(pos.getAsVector3df());

			core::vectorSIMDf up;
			auto target = pos;
			for (auto i=0; i<3; i++)
			{
				up[i] = relativeTransform.rows[i].y;
				target[i] += relativeTransform.rows[i].z;
			}

			if (relativeTransform.getPseudoDeterminant().x < 0.f)
				leftHandedCamera = true;

			camera->setTarget(target.getAsVector3df());
			camera->setUpVector(up.getAsVector3df());
		}

		const ext::MitsubaLoader::CElementSensor::PerspectivePinhole* persp = nullptr;
		switch (sensor.type)
		{
			case ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE:
				persp = &sensor.perspective;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::THINLENS:
				persp = &sensor.thinlens;
				break;
			default:
				assert(false);
				break;
		}
		float realFoVDegrees;
		auto width = viewport.getWidth();
		auto height = viewport.getHeight();
		float aspectRatio = float(width) / float(height);
		auto convertFromXFoV = [=](float fov) -> float
		{
			float aspectX = tan(core::radians(fov)*0.5f);
			return core::degrees(atan(aspectX/aspectRatio)*2.f);
		};
		switch (persp->fovAxis)
		{
			case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::X:
				realFoVDegrees = convertFromXFoV(persp->fov);
				break;
			case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::Y:
				realFoVDegrees = persp->fov;
				break;
			case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::DIAGONAL:
				{
					float aspectDiag = tan(core::radians(persp->fov)*0.5f);
					float aspectY = aspectDiag/core::sqrt(1.f+aspectRatio*aspectRatio);
					realFoVDegrees = core::degrees(atan(aspectY)*2.f);
				}
				break;
			case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::SMALLER:
				if (width < height)
					realFoVDegrees = convertFromXFoV(persp->fov);
				else
					realFoVDegrees = persp->fov;
				break;
			case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::LARGER:
				if (width < height)
					realFoVDegrees = persp->fov;
				else
					realFoVDegrees = convertFromXFoV(persp->fov);
				break;
			default:
				realFoVDegrees = NAN;
				assert(false);
				break;
		}
		if (leftHandedCamera)
			camera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(realFoVDegrees), aspectRatio, persp->nearClip, persp->farClip));
		else
			camera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(realFoVDegrees), aspectRatio, persp->nearClip, persp->farClip));
	}
	else
		camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
	camera->setLeftHanded(leftHandedCamera);
	smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);

	uint64_t lastFPSTime = 0;
	float lastFastestMeshFrameNr = -1.f;

	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));
		driver->setViewPort(viewport);

		//! This animates (moves) the camera and sets the transforms
		//! Also draws the meshbuffer
		smgr->drawAll();

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

	// create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	device->drop();
	return 0;
}
