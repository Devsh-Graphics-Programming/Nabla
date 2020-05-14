#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../ext/ScreenShot/ScreenShot.h"

#include "../common/QToQuitEventReceiver.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "../../ext/MitsubaLoader/CMitsubaLoader.h"

using namespace irr;
using namespace core;

/*
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
*/

#define TESTING
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
		auto device = createDeviceEx(params);


		io::IFileSystem* fs = device->getFileSystem();
		asset::IAssetManager* am = device->getAssetManager();
		asset::CQuantNormalCache* qnc = am->getMeshManipulator()->getQuantNormalCache();

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
		//qnc->loadNormalQuantCacheFromFile<asset::E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10>(fs, "../../tmp/normalCache101010.sse", true);
		//! load the mitsuba scene
		meshes = am->getAsset(filePath, {});
		//! cache results -- speeds up mesh generation on second run
		qnc->saveCacheToFile(asset::E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10, fs, "../../tmp/normalCache101010.sse");

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
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.


	scene::ISceneManager* smgr = device->getSceneManager();
	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);


	video::IVideoDriver* driver = device->getVideoDriver();

	core::vector<core::smart_refctd_ptr<asset::ICPUMesh>> cpumeshes;
	cpumeshes.reserve(meshes.getSize());
	for (auto it = meshes.getContents().first; it != meshes.getContents().second; ++it)
	{
		cpumeshes.push_back(core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(std::move(*it)));
	}
	core::smart_refctd_ptr<asset::ICPUDescriptorSet> cpuds0;
	{
		auto* meta = static_cast<ext::MitsubaLoader::CMitsubaPipelineMetadata*>(cpumeshes[0]->getMeshBuffer(0)->getPipeline()->getMetadata());
		cpuds0 = core::smart_refctd_ptr<asset::ICPUDescriptorSet>(meta->getDescriptorSet());
	}

	//all pipelines have the same metadata
	auto* pipelineMetadata = static_cast<const asset::IPipelineMetadata*>(cpumeshes.front()->getMeshBuffer(0u)->getPipeline()->getMetadata());

    asset::ICPUDescriptorSetLayout* ds1layout = cpumeshes.front()->getMeshBuffer(0u)->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
    uint32_t ds1UboBinding = 0u;
    for (const auto& bnd : ds1layout->getBindings())
        if (bnd.type==asset::EDT_UNIFORM_BUFFER)
        {
            ds1UboBinding = bnd.binding;
            break;
        }

    size_t neededDS1UBOsz = 0ull;
    {
        auto pipelineMetadata = static_cast<const asset::IPipelineMetadata*>(cpumeshes.front()->getMeshBuffer(0u)->getPipeline()->getMetadata());
        for (const auto& shdrIn : pipelineMetadata->getCommonRequiredInputs())
            if (shdrIn.descriptorSection.type==asset::IPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
                neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset+shdrIn.descriptorSection.uniformBufferObject.bytesize);
    }

	core::vector<const ext::MitsubaLoader::IMeshMetadata*> meshmetas;
	meshmetas.reserve(cpumeshes.size());
	for (const auto& cpumesh : cpumeshes)
		meshmetas.push_back(static_cast<const ext::MitsubaLoader::IMeshMetadata*>(cpumesh->getMetadata()));

	constexpr uint32_t MAX_INSTANCES = 512u;

	core::vector<core::smart_refctd_ptr<asset::ICPUBuffer>> cpuubos;
	cpuubos.reserve(cpumeshes.size());
	for (uint32_t i = 0u; i < cpumeshes.size(); ++i)
	{
		const uint32_t instCount = meshmetas[i]->getInstances().size();
		assert(instCount<=MAX_INSTANCES);
		auto ds = core::make_smart_refctd_ptr<asset::ICPUDescriptorSet>(core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(ds1layout));
		auto* info = ds->getDescriptors(ds1UboBinding).begin();
		auto buf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(core::matrix4SIMD)*instCount);
		info->buffer.offset = 0u;
		info->buffer.size = buf->getSize();
		info->desc = buf;

		for (uint32_t j = 0u; j < cpumeshes[i]->getMeshBufferCount(); ++j)
			cpumeshes[i]->getMeshBuffer(j)->setAttachedDescriptorSet(core::smart_refctd_ptr(ds));

		cpuubos.push_back(std::move(buf));
	}
	//we should have a way to access distinct descriptors from GPU DS
	auto gpuubos = driver->getGPUObjectsFromAssets(cpuubos.data(), cpuubos.data()+cpuubos.size());

	core::aabbox3df sceneBound;
	auto gpumeshes = driver->getGPUObjectsFromAssets(cpumeshes.data(), cpumeshes.data()+cpumeshes.size());
	{
		auto metait = meshmetas.begin();
		for (auto gpuit = gpumeshes->begin(); gpuit != gpumeshes->end(); gpuit++, metait++)
		{
			auto* meta = *metait;
			const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);
			const auto& instances = meshmeta->getInstances();

			auto bb = (*gpuit)->getBoundingBox();
			for (const auto& inst : instances)
			{
				sceneBound.addInternalBox(core::transformBoxEx(bb, inst.tform));
			}
		}
	}

	auto gpuds0 = driver->getGPUObjectsFromAssets(&cpuds0.get(), &cpuds0.get()+1)->front();

	// camera and viewport
	scene::ICameraSceneNode* camera = nullptr;
	core::recti viewport(core::position2di(0,0), core::position2di(params.WindowSize.Width,params.WindowSize.Height));

	auto isOkSensorType = [](const ext::MitsubaLoader::CElementSensor& sensor) -> bool {
		return sensor.type==ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE || sensor.type==ext::MitsubaLoader::CElementSensor::Type::THINLENS;
	};
#ifdef TESTING
	if (0)
#else
	if (globalMeta->sensors.size() && isOkSensorType(globalMeta->sensors.front()))
#endif
	{
		const auto& sensor = globalMeta->sensors.front();
		const auto& film = sensor.film;
		viewport = core::recti(core::position2di(film.cropOffsetX,film.cropOffsetY), core::position2di(film.cropWidth,film.cropHeight));

		auto extent = sceneBound.getExtent();
		camera = smgr->addCameraSceneNodeFPS(nullptr,100.f,core::min(extent.X,extent.Y,extent.Z)*0.0002f);
		// need to extract individual components
		bool leftHandedCamera = false;
		{
			auto relativeTransform = sensor.transform.matrix.extractSub3x4();
			if (relativeTransform.getPseudoDeterminant().x < 0.f)
				leftHandedCamera = true;

			auto pos = relativeTransform.getTranslation();
			camera->setPosition(pos.getAsVector3df());

			auto tpose = core::transpose(sensor.transform.matrix);
			auto up = tpose.rows[1];
			core::vectorSIMDf view = tpose.rows[2];
			auto target = view+pos;

			camera->setTarget(target.getAsVector3df());
			if (core::dot(core::normalize(core::cross(camera->getUpVector(),view)),core::cross(up,view)).x<0.99f)
				camera->setUpVector(up);
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
	smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);

	uint64_t lastFPSTime = 0;
	float lastFastestMeshFrameNr = -1.f;

	core::vector<core::matrix4SIMD> uboData(MAX_INSTANCES);
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));
		driver->setViewPort(viewport);
#ifdef TESTING
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();
#endif

		for (uint32_t j = 1u; j < gpumeshes->size(); ++j)
		{
			auto& mesh = (*gpumeshes)[j];
			auto* meta = meshmetas[j];
			const uint32_t instCount = meta->getInstances().size();
			{
				uint32_t i = 0u;
				for (auto& inst : meta->getInstances())
				{
					auto mvp = core::concatenateBFollowedByA(camera->getConcatenatedMatrix(), inst.tform);
					uboData[i++] = mvp;
				}
			}

			auto& ubo = (*gpuubos)[j];
			driver->updateBufferRangeViaStagingBuffer(ubo->getBuffer(), ubo->getOffset(), instCount*sizeof(core::matrix4SIMD), uboData.data());
			for (uint32_t i = 0u; i < mesh->getMeshBufferCount(); ++i)
			{
				auto* mb = mesh->getMeshBuffer(i);
				mb->setInstanceCount(instCount);

				auto* pipeline = mb->getPipeline();
				const auto* gpuds1 = mb->getAttachedDescriptorSet();
				const video::IGPUDescriptorSet* ds[2]{ gpuds0.get(), gpuds1 };
				driver->bindGraphicsPipeline(pipeline);
				driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 0u, 2u, ds, nullptr);

				driver->drawMeshBuffer(mb);
			}
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

	return 0;
}
