// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"

#define USE_ENVMAP

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

constexpr const char* GLSL_COMPUTE_LIGHTING =
R"(
#define _NBL_COMPUTE_LIGHTING_DEFINED_

#include <nbl/builtin/glsl/format/decode.glsl>
#include <nbl/builtin/glsl/random/xoroshiro.glsl>

struct SLight
{
	vec3 position;
	vec3 intensity;
};

layout (push_constant) uniform Block {
    float camTformDeterminant;
} PC;

layout (set = 2, binding = 0, std430) readonly restrict buffer Lights
{
	SLight lights[];
};
layout(set = 2, binding = 1) uniform sampler2D envMap;
layout(set = 2, binding = 2) uniform usamplerBuffer sampleSequence;
layout(set = 2, binding = 3) uniform usampler2D scramblebuf;

vec3 rand3d(in uint _sample, inout nbl_glsl_xoroshiro64star_state_t scramble_state)
{
	uvec3 seqVal = texelFetch(sampleSequence,int(_sample)).xyz;
	seqVal ^= uvec3(nbl_glsl_xoroshiro64star(scramble_state),nbl_glsl_xoroshiro64star(scramble_state),nbl_glsl_xoroshiro64star(scramble_state));
    return vec3(seqVal)*uintBitsToFloat(0x2f800004u);
}

vec2 SampleSphericalMap(in vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= nbl_glsl_RECIPROCAL_PI*0.5;
    uv += 0.5; 
    return uv;
}

vec3 nbl_computeLighting(inout nbl_glsl_AnisotropicViewSurfaceInteraction out_interaction)
{
	nbl_glsl_xoroshiro64star_state_t scramble_start_state = textureLod(scramblebuf,gl_FragCoord.xy/VIEWPORT_SZ,0).rg;

	vec3 emissive = nbl_glsl_MC_oriented_material_t_getEmissive(material);

	vec3 color = vec3(0.0);

#ifdef USE_ENVMAP
	nbl_glsl_MC_instr_stream_t gcs = nbl_glsl_MC_oriented_material_t_getGenChoiceStream(material);
	nbl_glsl_MC_instr_stream_t rnps = nbl_glsl_MC_oriented_material_t_getRemAndPdfStream(material);
	for (int i = 0; i < SAMPLE_COUNT; ++i)
	{
		nbl_glsl_xoroshiro64star_state_t scramble_state = scramble_start_state;

		vec3 rand = rand3d(i,scramble_state);
		float pdf;
		nbl_glsl_LightSample s;
		vec3 rem = nbl_glsl_MC_runGenerateAndRemainderStream(precomp, gcs, rnps, rand, pdf, s);

		vec2 uv = SampleSphericalMap(s.L);
		color += rem*textureLod(envMap, uv, 0.0).xyz;
	}
	color /= float(SAMPLE_COUNT);
#endif

	for (int i = 0; i < LIGHT_COUNT; ++i)
	{
		SLight l = lights[i];
		const vec3 L = l.position-WorldPos;
		const float d2 = dot(L,L); 
		const float intensityScale = LIGHT_INTENSITY_SCALE;//ehh might want to render to hdr fbo and do tonemapping
		nbl_glsl_LightSample _sample;
		_sample.L = L*inversesqrt(d2);
		color += nbl_bsdf_cos_eval(_sample,out_interaction)*l.intensity*intensityScale/d2;
	}

	return color+emissive;
}
)";
constexpr const char* GLSL_FRAG_MAIN = R"(
#define _NBL_FRAG_MAIN_DEFINED_
void main()
{
	mat2 dUV = mat2(dFdx(UV),dFdy(UV));

	// "The sign of this computation is negated when the value of GL_CLIP_ORIGIN (the clip volume origin, set with glClipControl) is GL_UPPER_LEFT."
	const bool front = bool((InstData.data[InstanceIndex].determinantSignBit^mix(~0u,0u,gl_FrontFacing!=PC.camTformDeterminant<0.0))&0x80000000u);
	precomp = nbl_glsl_MC_precomputeData(front);
	material = nbl_glsl_MC_material_data_t_getOriented(InstData.data[InstanceIndex].material,precomp.frontface);
#ifdef TEX_PREFETCH_STREAM
	nbl_glsl_MC_runTexPrefetchStream(nbl_glsl_MC_oriented_material_t_getTexPrefetchStream(material), UV, dUV);
#endif
#ifdef NORM_PRECOMP_STREAM
	nbl_glsl_MC_runNormalPrecompStream(nbl_glsl_MC_oriented_material_t_getNormalPrecompStream(material), precomp);
#endif


	nbl_glsl_AnisotropicViewSurfaceInteraction inter;
	vec3 color = nbl_computeLighting(inter);

	OutColor = vec4(color, 1.0);
}
)";

static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createModifiedFragShader(const asset::ICPUSpecializedShader* _fs, uint32_t viewport_w, uint32_t viewport_h, uint32_t lightCnt, uint32_t smplCnt, float intensityScale)
{
	const asset::ICPUShader* unspec = _fs->getUnspecialized();
	assert(unspec->containsGLSL());

	std::string glsl = reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer());
	std::string extra = "\n#define VIEWPORT_SZ vec2(" + std::to_string(viewport_w) + "," + std::to_string(viewport_h) + ")" +
		"\n#define LIGHT_COUNT " + std::to_string(lightCnt) +
		"\n#define SAMPLE_COUNT " + std::to_string(smplCnt) +
		"\n#define LIGHT_INTENSITY_SCALE " + std::to_string(intensityScale) +
#ifdef USE_ENVMAP
		"\n#define USE_ENVMAP" +
#endif
		GLSL_COMPUTE_LIGHTING;

	glsl.insert(glsl.find("#ifndef _NBL_COMPUTE_LIGHTING_DEFINED_"), extra);
	glsl.insert(glsl.find("#ifndef _NBL_FRAG_MAIN_DEFINED_"), GLSL_FRAG_MAIN);

	//auto* f = fopen("fs.glsl","w");
	//fwrite(glsl.c_str(), 1, glsl.size(), f);
	//fclose(f);

	auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(glsl.c_str());
	auto specinfo = _fs->getSpecializationInfo();//intentional copy
	auto fsNew = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));

	return fsNew;
}

static auto createGPUImageView(const std::string& path, IAssetManager* am, ILogicalDevice* logicalDevice, nbl::video::IGPUObjectFromAssetConverter& cpu2gpu, nbl::video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams)
{
	IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
	auto cpuTexture = am->getAsset(path, lp);
	auto cpuTextureContents = cpuTexture.getContents();

	auto asset = *cpuTextureContents.begin();
	auto cpuimg = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);

	IGPUImageView::SCreationParams viewParams;
	viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
	viewParams.image = cpu2gpu.getGPUObjectsFromAssets(&cpuimg.get(), &cpuimg.get() + 1u, cpu2gpuParams)->begin()[0];
	cpu2gpuParams.waitForCreationToComplete();
	viewParams.format = viewParams.image->getCreationParameters().format;
	viewParams.viewType = IImageView<IGPUImage>::ET_2D;
	viewParams.subresourceRange.baseArrayLayer = 0u;
	viewParams.subresourceRange.layerCount = 1u;
	viewParams.subresourceRange.baseMipLevel = 0u;
	viewParams.subresourceRange.levelCount = viewParams.image->getCreationParameters().mipLevels;

	auto gpuImageView = logicalDevice->createGPUImageView(std::move(viewParams));

	return gpuImageView;
};

#include "nbl/nblpack.h"
//std430-compatible
struct SLight
{
	core::vectorSIMDf position;
	core::vectorSIMDf intensity;
} PACK_STRUCT;
#include "nbl/nblunpack.h"

int main(int argc, char** argv)
{
	system::path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
	constexpr uint32_t WIN_W = 1024;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	auto initOutput = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "MitsubaLoader", nbl::asset::EF_D32_SFLOAT);
	auto window = std::move(initOutput.window);
	auto gl = std::move(initOutput.apiConnection);
	auto surface = std::move(initOutput.surface);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto logicalDevice = std::move(initOutput.logicalDevice);
	auto queues = std::move(initOutput.queues);
	auto swapchain = std::move(initOutput.swapchain);
	auto renderpass = std::move(initOutput.renderpass);
	auto fbos = std::move(initOutput.fbo);
	auto commandPool = std::move(initOutput.commandPool);
	auto assetManager = std::move(initOutput.assetManager);
	auto logger = std::move(initOutput.logger);
	auto inputSystem = std::move(initOutput.inputSystem);
	auto system = std::move(initOutput.system);
	auto windowCallback = std::move(initOutput.windowCb);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	auto utilities = std::move(initOutput.utilities);

	core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
	core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
	{
		cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
		cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
	}

	// Select mitsuba file
	asset::SAssetBundle meshes;
	core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata> globalMeta;
	{
		asset::CQuantNormalCache* qnc = assetManager->getMeshManipulator()->getQuantNormalCache();

		auto serializedLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CSerializedLoader>(assetManager.get());
		auto mitsubaLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CMitsubaLoader>(assetManager.get(), system.get());
		serializedLoader->initialize();
		mitsubaLoader->initialize();
		assetManager->addAssetLoader(std::move(serializedLoader));
		assetManager->addAssetLoader(std::move(mitsubaLoader));

		std::string filePath = "../../media/mitsuba/bathroom.zip";
		system::path parentPath;
		#define MITSUBA_LOADER_TESTS
#ifndef MITSUBA_LOADER_TESTS
		pfd::message("Choose file to load", "Choose mitsuba XML file to load or ZIP containing an XML. \nIf you cancel or choosen file fails to load staircase will be loaded.", pfd::choice::ok);
		pfd::open_file file("Choose XML or ZIP file", (CWD/"../../media/mitsuba").string(), { "ZIP files (.zip)", "*.zip", "XML files (.xml)", "*.xml" });
		if (!file.result().empty())
			filePath = file.result()[0];
#endif
		if (core::hasFileExtension(filePath, "zip", "ZIP"))
		{
			const system::path archPath = CWD/filePath;
			core::smart_refctd_ptr<system::IFileArchive> arch = nullptr;
			arch = system->openFileArchive(archPath);

			if (!arch)
				arch = system->openFileArchive(CWD/ "../../media/mitsuba/bathroom.zip");
			if (!arch)
				return 2;

			system->mount(std::move(arch), "resources");

			auto flist = arch->getArchivedFiles();
			if (flist.empty())
				return 3;

			for (auto it = flist.begin(); it != flist.end(); )
			{
				if (core::hasFileExtension(it->fullName, "xml", "XML"))
					it++;
				else
					it = flist.erase(it);
			}
			if (flist.size() == 0u)
				return 4;

			std::cout << "Choose File (0-" << flist.size() - 1ull << "):" << std::endl;
			for (auto i = 0u; i < flist.size(); i++)
				std::cout << i << ": " << flist[i].fullName << std::endl;
			uint32_t chosen = 0;
#ifndef MITSUBA_LOADER_TESTS
			std::cin >> chosen;
#endif
			if (chosen >= flist.size())
				chosen = 0u;

			filePath = flist[chosen].name.string();
			parentPath = flist[chosen].fullName.parent_path();
		}

		//! read cache results -- speeds up mesh generation
		qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");
		//! load the mitsuba scene
		asset::IAssetLoader::SAssetLoadParams loadParams;
		loadParams.workingDirectory = "resources"/parentPath;
		loadParams.logger = logger.get();
		meshes = assetManager->getAsset(filePath, loadParams);
		assert(!meshes.getContents().empty());
		//! cache results -- speeds up mesh generation on second run
		qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");

		auto contents = meshes.getContents();
		if (contents.begin() >= contents.end())
			return 2;

		auto firstmesh = *contents.begin();
		if (!firstmesh)
			return 3;

		globalMeta = core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata>(meshes.getMetadata()->selfCast<const ext::MitsubaLoader::CMitsubaMetadata>());
		if (!globalMeta)
			return 4;
	}

	// gather all meshes into core::vector and modify their pipelines

	core::vector<core::smart_refctd_ptr<asset::ICPUMesh>> cpuMeshes;
	asset::ICPUDescriptorSetLayout* ds1Layout;
	core::smart_refctd_ptr<ICPUDescriptorSet> cpuDS0 = globalMeta->m_global.m_ds0;
	uint32_t ds1UboBinding = 0u;
	{
		auto contents = meshes.getContents();

		cpuMeshes.reserve(contents.size());
		for (auto it = contents.begin(); it != contents.end(); ++it)
			cpuMeshes.push_back(core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(std::move(*it)));

		ds1Layout = cpuMeshes.front()->getMeshBuffers().begin()[0u]->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
		for (const auto& bnd : ds1Layout->getBindings())
		{
			if (bnd.type == asset::EDT_UNIFORM_BUFFER)
			{
				ds1UboBinding = bnd.binding;
				break;
			}
		}
	}

	// process metadata

		// process sensor

	const auto& sensor = globalMeta->m_global.m_sensors.front(); //always choose frist one
	auto isOkSensorType = [](const ext::MitsubaLoader::CElementSensor& sensor) -> bool {
		return sensor.type == ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE || sensor.type == ext::MitsubaLoader::CElementSensor::Type::THINLENS;
	};

	if (!isOkSensorType(sensor))
		return 1;

	bool leftHandedCamera = false;
	auto cameraTransform = sensor.transform.matrix.extractSub3x4();
	{
		if (cameraTransform.getPseudoDeterminant().x < 0.f)
			leftHandedCamera = true;
	}

		// process lights

	//scene bound
	core::aabbox3df sceneBound(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
	//point lights
	core::vector<SLight> lights;
	for (const auto& cpuMesh : cpuMeshes)
	{
		auto computeAreaAndAvgPos = [](const asset::ICPUMeshBuffer* mb, const core::matrix3x4SIMD& tform, core::vectorSIMDf& _outAvgPos) {
			uint32_t triCount = 0u;
			asset::IMeshManipulator::getPolyCount(triCount, mb);
			assert(triCount > 0u);

			_outAvgPos = core::vectorSIMDf(0.f);

			core::vectorSIMDf v[3];
			for (uint32_t i = 0u; i < triCount; ++i)
			{
				auto triIxs = asset::IMeshManipulator::getTriangleIndices(mb, i);
				for (uint32_t j = 0u; j < 3u; ++j) {
					v[j] = mb->getPosition(triIxs[j]);
					_outAvgPos += v[j];
				}
			}
			core::vectorSIMDf differentialElementCrossProdcut = core::cross(v[1] - v[0], v[2] - v[0]);
			core::matrix3x4SIMD tformCofactors;
			{
				auto tmp4 = nbl::core::matrix4SIMD(tform.getSub3x3TransposeCofactors());
				tformCofactors = core::transpose(tmp4).extractSub3x4();
			}
			tformCofactors.mulSub3x3WithNx1(differentialElementCrossProdcut);

			//outputs position in model space
			_outAvgPos /= static_cast<float>(triCount) * 3.f;

			return 0.5f * core::length(differentialElementCrossProdcut).x;
		};

		const auto* mesh_meta = globalMeta->getAssetSpecificMetadata(cpuMesh.get());
		auto auxInstanceDataIt = mesh_meta->m_instanceAuxData.begin();
		for (const auto& inst : mesh_meta->m_instances)
		{
			sceneBound.addInternalBox(core::transformBoxEx(cpuMesh->getBoundingBox(), inst.worldTform));
			if (auxInstanceDataIt->frontEmitter.type == ext::MitsubaLoader::CElementEmitter::AREA)
			{
				core::vectorSIMDf pos;
				assert(cpuMesh->getMeshBuffers().size() == 1u);
				const float area = computeAreaAndAvgPos(cpuMesh->getMeshBuffers().begin()[0], inst.worldTform, pos);
				assert(area > 0.f);
				inst.worldTform.pseudoMulWith4x1(pos);

				SLight l;
				l.intensity = auxInstanceDataIt->frontEmitter.area.radiance * area * 2.f * core::PI<float>();
				l.position = pos;

				lights.push_back(l);
			}
			auxInstanceDataIt++;
		}
	}

	// process sensor

	core::recti viewport(core::position2di(0, 0), core::position2di(WIN_W, WIN_H));

	core::vectorSIMDf cameraPosition(0, 5, -10);
	matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.1, 1000);
	core::vectorSIMDf lookAt(0, 0, 0);
	core::vectorSIMDf upVec(0, 1, 0);
	float moveSpeed = 10.0f;

	//#define TESTING
#ifdef TESTING
	if (0)
#else
	if (globalMeta->m_global.m_sensors.size() && isOkSensorType(globalMeta->m_global.m_sensors.front()))
#endif
	{
		core::vectorSIMDf cameraPositionRef(0, 5, -10);
		matrix4SIMD projectionMatrixRef = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.1, 1000);

		const auto& film = sensor.film;
		viewport = core::recti(core::position2di(film.cropOffsetX, film.cropOffsetY), core::position2di(film.cropWidth, film.cropHeight));

		auto extent = sceneBound.getExtent();
		moveSpeed = core::min(extent.X, extent.Y, extent.Z) * 0.01f;

		// need to extract individual components
		{
			auto relativeTransform = sensor.transform.matrix.extractSub3x4();

			const auto pos = relativeTransform.getTranslation();
			cameraPosition = pos;

			auto tpose = core::transpose(sensor.transform.matrix);
			auto up = tpose.rows[1];
			core::vectorSIMDf view = tpose.rows[2];
			auto target = view + pos;

			lookAt = target;


			if (core::dot(core::normalize(core::cross(upVec, view)), core::cross(up, view)).x < 0.99f)
				upVec = up;
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
			float aspectX = tan(core::radians(fov) * 0.5f);
			return core::degrees(atan(aspectX / aspectRatio) * 2.f);
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
			float aspectDiag = tan(core::radians(persp->fov) * 0.5f);
			float aspectY = aspectDiag / core::sqrt(1.f + aspectRatio * aspectRatio);
			realFoVDegrees = core::degrees(atan(aspectY) * 2.f);
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
			projectionMatrix = core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(realFoVDegrees), aspectRatio, persp->nearClip, persp->farClip);
		else
			projectionMatrix = core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(realFoVDegrees), aspectRatio, persp->nearClip, persp->farClip);
	}

	Camera camera = Camera(cameraPosition, lookAt, projectionMatrix, moveSpeed, 1.f, upVec);

	// convert cpu meshes to gpu meshes
	

	//TODO:
	//// recreate wth resolution
	//params.WindowSize = dimension2d<uint32_t>(1280, 720);
	//// set resolution
	//if (globalMeta->m_global.m_sensors.size())
	//{
	//	const auto& film = globalMeta->m_global.m_sensors.front().film;
	//	params.WindowSize.Width = film.width;
	//	params.WindowSize.Height = film.height;
	//}
	//else return 1; // no cameras

	core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds2Layout;
	{
		asset::ICPUDescriptorSetLayout::SBinding bnd[4];
		bnd[0].binding = 0u;
		bnd[0].count = 1u;
		bnd[0].samplers = nullptr;
		bnd[0].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
		bnd[0].type = asset::EDT_STORAGE_BUFFER;

		using namespace asset;
		ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		auto smplr = core::make_smart_refctd_ptr<asset::ICPUSampler>(samplerParams);
		bnd[1].binding = 1u;
		bnd[1].count = 1u;
		bnd[1].samplers = &smplr;
		bnd[1].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
		bnd[1].type = asset::EDT_COMBINED_IMAGE_SAMPLER;

		bnd[2].binding = 2u;
		bnd[2].count = 1u;
		bnd[2].samplers = nullptr;
		bnd[2].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
		bnd[2].type = asset::EDT_UNIFORM_TEXEL_BUFFER;

		samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
		auto smplr_int = core::make_smart_refctd_ptr<asset::ICPUSampler>(samplerParams);
		bnd[3].binding = 3u;
		bnd[3].count = 1u;
		bnd[3].samplers = &smplr_int;
		bnd[3].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
		bnd[3].type = asset::EDT_COMBINED_IMAGE_SAMPLER;

		ds2Layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bnd, bnd + 4);
	}

	constexpr uint32_t ENVMAP_SAMPLE_COUNT = 64u;
	constexpr float LIGHT_INTENSITY_SCALE = 0.01f;

	core::unordered_set<const asset::ICPURenderpassIndependentPipeline*> modifiedPipelines;
	core::unordered_map<core::smart_refctd_ptr<asset::ICPUSpecializedShader>, core::smart_refctd_ptr<asset::ICPUSpecializedShader>> modifiedShaders;
	for (auto& mesh : cpuMeshes)
	{
		//modify pipeline layouts with our custom DS2 layout (DS2 will be used for lights buffer)
		for (auto meshbuffer : mesh->getMeshBuffers())
		{
			auto* pipeline = meshbuffer->getPipeline();

			asset::SPushConstantRange pcr;
			pcr.offset = 0u;
			pcr.size = sizeof(float);
			pcr.stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
			if (modifiedPipelines.find(pipeline) == modifiedPipelines.end())
			{
				//if (!pipeline->getLayout()->getDescriptorSetLayout(2u))
				pipeline->getLayout()->setDescriptorSetLayout(2u, core::smart_refctd_ptr(ds2Layout));
				auto* fs = pipeline->getShaderAtStage(asset::ICPUSpecializedShader::ESS_FRAGMENT);
				auto found = modifiedShaders.find(core::smart_refctd_ptr<asset::ICPUSpecializedShader>(fs));
				if (found != modifiedShaders.end())
					pipeline->setShaderAtStage(asset::ICPUSpecializedShader::ESS_FRAGMENT, found->second.get());
				else {
					auto newfs = createModifiedFragShader(fs, WIN_W, WIN_H, lights.size(), ENVMAP_SAMPLE_COUNT, LIGHT_INTENSITY_SCALE);
					modifiedShaders.insert({ core::smart_refctd_ptr<asset::ICPUSpecializedShader>(fs),newfs });
					pipeline->setShaderAtStage(asset::ICPUSpecializedShader::ESS_FRAGMENT, newfs.get());
				}

				auto pc = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::SPushConstantRange>>(1u);
				(*pc)[0] = pcr;
				pipeline->getLayout()->setPushConstantRanges(std::move(pc));
				modifiedPipelines.insert(pipeline);
			}

			reinterpret_cast<float*>(meshbuffer->getPushConstantsDataPtr() + pcr.offset)[0] = cameraTransform.getPseudoDeterminant().x;
		}
	}
	modifiedShaders.clear();

	// create gpu pipelines

	auto gpuMeshes = cpu2gpu.getGPUObjectsFromAssets(cpuMeshes.data(), cpuMeshes.data() + cpuMeshes.size(), cpu2gpuParams);
	cpu2gpuParams.waitForCreationToComplete();

	auto gpuDS0 = cpu2gpu.getGPUObjectsFromAssets(&cpuDS0.get(), &cpuDS0.get() + 1, cpu2gpuParams)->begin()[0];
	cpu2gpuParams.waitForCreationToComplete();
	auto gpuDS1Layout = cpu2gpu.getGPUObjectsFromAssets(&ds1Layout, &ds1Layout + 1, cpu2gpuParams)->begin()[0];
	cpu2gpuParams.waitForCreationToComplete();
	auto gpuDS2Layout = cpu2gpu.getGPUObjectsFromAssets(&ds2Layout, &ds2Layout + 1, cpu2gpuParams)->front();
	cpu2gpuParams.waitForCreationToComplete();

	IGPUDescriptorSetLayout* gpuLayouts[2] = {
		gpuDS1Layout.get(), gpuDS2Layout.get()
	};

	core::smart_refctd_ptr<IDescriptorPool> descriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, gpuLayouts, gpuLayouts + 2);

	core::vector<core::smart_refctd_ptr<IGPUGraphicsPipeline>> graphicsPplns;
	IDriverMemoryBacked::SDriverMemoryRequirements memReq;
	memReq.vulkanReqs.size = sizeof(SBasicViewParameters);
	core::smart_refctd_ptr<IGPUBuffer> cameraUBO = logicalDevice->createGPUBufferOnDedMem(memReq, true);
	auto gpuDS1 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), std::move(gpuDS1Layout));
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write;
		write.dstSet = gpuDS1.get();
		write.binding = ds1UboBinding;
		write.count = 1u;
		write.arrayElement = 0u;
		write.descriptorType = asset::EDT_UNIFORM_BUFFER;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = cameraUBO;
			info.buffer.offset = 0ull;
			info.buffer.size = cameraUBO->getSize();
		}
		write.info = &info;
		logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
	}

	smart_refctd_ptr<video::IGPUBufferView> gpuSequenceBufferView;
	{
		constexpr uint32_t MaxSamples = ENVMAP_SAMPLE_COUNT;
		constexpr uint32_t Channels = 3u;

		auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t) * MaxSamples * Channels);

		core::OwenSampler sampler(Channels, 0xdeadbeefu);

		auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
		for (uint32_t c = 0; c < Channels; ++c)
			for (uint32_t i = 0; i < MaxSamples; i++)
			{
				out[Channels * i + c] = sampler.sample(c, i);
			}

		auto gpuSSBOOffsetBufferPair = cpu2gpu.getGPUObjectsFromAssets(&sampleSequence, &sampleSequence + 1u, cpu2gpuParams)->begin()[0];
		cpu2gpuParams.waitForCreationToComplete();
		auto gpuSequenceBuffer = core::smart_refctd_ptr<video::IGPUBuffer>(gpuSSBOOffsetBufferPair->getBuffer());

		gpuSequenceBufferView = logicalDevice->createGPUBufferView(gpuSequenceBuffer.get(), asset::EF_R32G32B32_UINT);
	}

	smart_refctd_ptr<video::IGPUImageView> gpuScrambleImageView;
	{
		ICPUImage::SCreationParams imgParams;
		imgParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
		imgParams.type = asset::IImage::ET_2D;
		imgParams.format = asset::EF_R32G32_UINT;
		imgParams.extent = { WIN_W, WIN_H, 1u };
		imgParams.mipLevels = 1u;
		imgParams.arrayLayers = 1u;
		imgParams.samples = asset::IImage::ESCF_1_BIT;

		core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy> regionArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
		auto& region = regionArray->begin()[0];
		region.imageExtent = imgParams.extent;
		region.imageSubresource.layerCount = 1u;

		constexpr auto ScrambleStateChannels = 2u;
		const auto renderPixelCount = imgParams.extent.width * imgParams.extent.height;
		core::vector<uint32_t> random(renderPixelCount * ScrambleStateChannels);
		{
			core::RandomSampler rng(0xbadc0ffeu);
			for (auto& pixel : random)
				pixel = rng.nextSample();
		}

		auto cpuBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(random.size() * sizeof(uint32_t));
		memcpy(cpuBuffer->getPointer(), random.data(), cpuBuffer->getSize());
		auto gpuSSBOOffsetBufferPair = cpu2gpu.getGPUObjectsFromAssets(&cpuBuffer, &cpuBuffer + 1u, cpu2gpuParams)->begin()[0];
		cpu2gpuParams.waitForCreationToComplete();
		auto buffer = core::smart_refctd_ptr<video::IGPUBuffer>(gpuSSBOOffsetBufferPair->getBuffer());

		core::smart_refctd_ptr<ICPUImage> cpuImage = ICPUImage::create(std::move(imgParams));
		cpuImage->setBufferAndRegions(std::move(cpuBuffer), regionArray);

		video::IGPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = cpu2gpu.getGPUObjectsFromAssets(&cpuImage, &cpuImage + 1u, cpu2gpuParams)->begin()[0];
		cpu2gpuParams.waitForCreationToComplete();
		viewParams.viewType = video::IGPUImageView::ET_2D;
		viewParams.format = asset::EF_R32G32_UINT;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.layerCount = 1u;
		gpuScrambleImageView = logicalDevice->createGPUImageView(std::move(viewParams));
	}

	auto gpuDS2 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), std::move(gpuDS2Layout));
	{
		auto cpuBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(lights.size() * sizeof(SLight));
		memcpy(cpuBuffer->getPointer(), lights.data(), cpuBuffer->getSize());
		auto gpuSSBOOffsetBufferPair = cpu2gpu.getGPUObjectsFromAssets(&cpuBuffer, &cpuBuffer + 1u, cpu2gpuParams)->begin()[0];
		cpu2gpuParams.waitForCreationToComplete();
		auto lightsBuf = core::smart_refctd_ptr<video::IGPUBuffer>(gpuSSBOOffsetBufferPair->getBuffer());

		video::IGPUDescriptorSet::SDescriptorInfo info[4];
		video::IGPUDescriptorSet::SWriteDescriptorSet w[4];
		w[0].arrayElement = 0u;
		w[0].binding = 0u;
		w[0].count = 1u;
		w[0].descriptorType = asset::EDT_STORAGE_BUFFER;
		w[0].dstSet = gpuDS2.get();
		w[0].info = info + 0;
		info[0].buffer.offset = 0u;
		info[0].buffer.size = lightsBuf->getSize();
		info[0].desc = std::move(lightsBuf);

		w[1].arrayElement = 0u;
		w[1].binding = 1u;
		w[1].count = 1u;
		w[1].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
		w[1].dstSet = gpuDS2.get();
		w[1].info = info + 1;
		auto gpuEnvmapImageView = createGPUImageView("../../media/envmap/envmap_0.exr", assetManager.get(), logicalDevice.get(), cpu2gpu, cpu2gpuParams);
		info[1].image.imageLayout = asset::EIL_UNDEFINED;
		info[1].image.sampler = nullptr;
		info[1].desc = std::move(gpuEnvmapImageView);

		w[2].arrayElement = 0u;
		w[2].binding = 2u;
		w[2].count = 1u;
		w[2].descriptorType = asset::EDT_UNIFORM_TEXEL_BUFFER;
		w[2].dstSet = gpuDS2.get();
		w[2].info = info + 2;
		info[2].desc = gpuSequenceBufferView;

		w[3].arrayElement = 0u;
		w[3].binding = 3u;
		w[3].count = 1u;
		w[3].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
		w[3].dstSet = gpuDS2.get();
		w[3].info = info + 3;
		info[3].image.imageLayout = asset::EIL_UNDEFINED;
		info[3].image.sampler = nullptr;//imm sampler is present
		info[3].desc = gpuScrambleImageView;

		logicalDevice->updateDescriptorSets(4u, w, 0u, nullptr);

	}

	for (uint32_t i = 0u; i < gpuMeshes->size(); i++)
	{
		auto& mesh = (*gpuMeshes)[i];

		for (auto mb : mesh->getMeshBuffers())
		{
			nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
			graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(mb->getPipeline()));
			graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

			graphicsPplns.emplace_back(logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams)));
		}
	}

	// setup

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	auto lastTime = std::chrono::system_clock::now();

	constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;
	bool frameDataFilled = false;
	size_t frame_count = 0ull;
	double time_sum = 0;
	double dtList[NBL_FRAMES_TO_AVERAGE] = {};
	for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
		dtList[i] = 0.0;

	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
	logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
	{
		imageAcquire[i] = logicalDevice->createSemaphore();
		renderFinished[i] = logicalDevice->createSemaphore();
	}

	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	uint32_t acquiredNextFBO = {};
	auto resourceIx = -1;
	
	float lastFastestMeshFrameNr = -1.f;

	// MAIN LOOP
	while (windowCallback->isWindowOpen())
	{
		++resourceIx;
		if (resourceIx >= FRAMES_IN_FLIGHT)
			resourceIx = 0;

		auto& commandBuffer = commandBuffers[resourceIx];
		auto& fence = frameComplete[resourceIx];

		if (fence)
			while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		auto renderStart = std::chrono::system_clock::now();
		const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - lastTime).count();
		lastTime = renderStart;
		{ // Calculate Simple Moving Average for FrameTime
			time_sum -= dtList[frame_count];
			time_sum += renderDt;
			dtList[frame_count] = renderDt;
			frame_count++;
			if (frame_count >= NBL_FRAMES_TO_AVERAGE)
			{
				frameDataFilled = true;
				frame_count = 0;
			}

		}
		const double averageFrameTime = frameDataFilled ? (time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (time_sum / frame_count);

#ifdef NBL_MORE_LOGS
		logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
#endif // NBL_MORE_LOGS

		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		camera.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
		camera.endInputProcessing(nextPresentationTimeStamp);

		const auto& viewMatrix = camera.getViewMatrix();
		const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

		commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
		commandBuffer->begin(0);

		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		commandBuffer->setViewport(0u, 1u, &viewport);

		swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { WIN_W, WIN_H };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 1.f;
			clear[0].color.float32[1] = 1.f;
			clear[0].color.float32[2] = 1.f;
			clear[0].color.float32[3] = 1.f;
			clear[1].depthStencil.depth = 0.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = fbos[acquiredNextFBO];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
		
		// draw shit
		
		SBasicViewParameters viewParams;
		std::memcpy(viewParams.MVP, viewProjectionMatrix.pointer(), sizeof(core::matrix4SIMD));
		std::memcpy(viewParams.MV, viewMatrix.pointer(), sizeof(core::matrix3x4SIMD));
		commandBuffer->updateBuffer(cameraUBO.get(), 0ull, cameraUBO->getSize(), &viewParams) == false;
		for (uint32_t i = 0u, mbIdx = 0; i < gpuMeshes->size(); i++, mbIdx++)
		{
			auto& mesh = (*gpuMeshes)[i];

			for (auto mb : mesh->getMeshBuffers())
			{
				auto* pipeline = mb->getPipeline();
				const video::IGPUDescriptorSet* ds[3]{ gpuDS0.get(), gpuDS1.get(), gpuDS2.get() };

				const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = mb->getPipeline();

				commandBuffer->bindGraphicsPipeline(graphicsPplns[mbIdx].get());

				// TODO: different approach?
				mbIdx++;

				commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 0u, 3u, ds, nullptr);

				commandBuffer->drawMeshBuffer(mb);
			}
		}

		commandBuffer->endRenderPass();
		commandBuffer->end();

		CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
	}
}