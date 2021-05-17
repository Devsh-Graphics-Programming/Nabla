// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "../common/QToQuitEventReceiver.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"
#include <nbl/video/IGPUVirtualTexture.h>

#define USE_ENVMAP

using namespace nbl;
using namespace core;

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

    std::string glsl = reinterpret_cast<const char*>( unspec->getSPVorGLSL()->getPointer() );
	std::string extra = "\n#define VIEWPORT_SZ vec2(" + std::to_string(viewport_w)+","+std::to_string(viewport_h)+")" +
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

static auto createGPUImageView(const std::string& path, asset::IAssetManager* am, video::IVideoDriver* driver)
{
	using namespace asset;
	using namespace video;

	IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
	auto cpuTexture = am->getAsset(path, lp);
	auto cpuTextureContents = cpuTexture.getContents();

	auto asset = *cpuTextureContents.begin();
	auto cpuimg = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);

	IGPUImageView::SCreationParams viewParams;
	viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
	viewParams.image = driver->getGPUObjectsFromAssets(&cpuimg.get(),&cpuimg.get()+1u)->front();
	viewParams.format = viewParams.image->getCreationParameters().format;
	viewParams.viewType = IImageView<IGPUImage>::ET_2D;
	viewParams.subresourceRange.baseArrayLayer = 0u;
	viewParams.subresourceRange.layerCount = 1u;
	viewParams.subresourceRange.baseMipLevel = 0u;
	viewParams.subresourceRange.levelCount = viewParams.image->getCreationParameters().mipLevels;

	auto gpuImageView = driver->createGPUImageView(std::move(viewParams));

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

int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_NULL;
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon

	//
	asset::SAssetBundle meshes;
	core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata> globalMeta;
	{
		auto device = createDeviceEx(params);


		io::IFileSystem* fs = device->getFileSystem();
		asset::IAssetManager* am = device->getAssetManager();
		asset::CQuantNormalCache* qnc = am->getMeshManipulator()->getQuantNormalCache();

		auto serializedLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CSerializedLoader>(am);
		auto mitsubaLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CMitsubaLoader>(am,fs);
		serializedLoader->initialize();
		mitsubaLoader->initialize();
		am->addAssetLoader(std::move(serializedLoader));
		am->addAssetLoader(std::move(mitsubaLoader));

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
		qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");
		//! load the mitsuba scene
		meshes = am->getAsset(filePath, {});
		//! cache results -- speeds up mesh generation on second run
		qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");
		
		auto contents = meshes.getContents();
		if (contents.begin()>=contents.end())
			return 2;

		auto firstmesh = *contents.begin();
		if (!firstmesh)
			return 3;

		globalMeta = core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata>(meshes.getMetadata()->selfCast<const ext::MitsubaLoader::CMitsubaMetadata>());
		if (!globalMeta)
			return 4;
	}


	// recreate wth resolution
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	// set resolution
	if (globalMeta->m_global.m_sensors.size())
	{
		const auto& film = globalMeta->m_global.m_sensors.front().film;
		params.WindowSize.Width = film.width;
		params.WindowSize.Height = film.height;
	}
	else return 1; // no cameras

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

	params.DriverType = video::EDT_OPENGL;
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.


	scene::ISceneManager* smgr = device->getSceneManager();
	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);


	video::IVideoDriver* driver = device->getVideoDriver();
	asset::IAssetManager* am = device->getAssetManager();
	io::IFileSystem* fs = device->getFileSystem();

	// look out for this!!!
	// when added, CMitsubaLoader inserts its own include loader into GLSLCompiler
	// thats why i have to add it again here (after device recreation) to be able to compile shaders
	{
		auto mitsubaLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CMitsubaLoader>(am, fs);
		mitsubaLoader->initialize();
		am->addAssetLoader(std::move(mitsubaLoader));
	}

	core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds2layout;
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

		ds2layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bnd, bnd + 4);
	}

	auto contents = meshes.getContents();
	//gather all meshes into core::vector and modify their pipelines
	core::vector<core::smart_refctd_ptr<asset::ICPUMesh>> cpumeshes;
	cpumeshes.reserve(contents.size());
	uint32_t cc = cpumeshes.capacity();
	for (auto it=contents.begin(); it!=contents.end(); ++it)
		cpumeshes.push_back(core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(std::move(*it)));

	auto cpuds0 = globalMeta->m_global.m_ds0;

    asset::ICPUDescriptorSetLayout* ds1layout = cpumeshes.front()->getMeshBuffers().begin()[0u]->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
    uint32_t ds1UboBinding = 0u;
    for (const auto& bnd : ds1layout->getBindings())
    if (bnd.type==asset::EDT_UNIFORM_BUFFER)
    {
        ds1UboBinding = bnd.binding;
        break;
    }

	//scene bound
	core::aabbox3df sceneBound(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX);
	//point lights
	core::vector<SLight> lights;
	for (const auto& cpumesh : cpumeshes)
	{
		auto computeAreaAndAvgPos = [](const asset::ICPUMeshBuffer* mb, const core::matrix3x4SIMD& tform, core::vectorSIMDf& _outAvgPos) {
			uint32_t triCount = 0u;
			asset::IMeshManipulator::getPolyCount(triCount, mb);
			assert(triCount>0u);

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
			core::vectorSIMDf differentialElementCrossProdcut = core::cross(v[1]-v[0], v[2]-v[0]);
			core::matrix3x4SIMD tformCofactors;
			{
				auto tmp4 = nbl::core::matrix4SIMD(tform.getSub3x3TransposeCofactors());
				tformCofactors = core::transpose(tmp4).extractSub3x4();
			}
			tformCofactors.mulSub3x3WithNx1(differentialElementCrossProdcut);

			//outputs position in model space
			_outAvgPos /= static_cast<float>(triCount)*3.f;

			return 0.5f*core::length(differentialElementCrossProdcut).x;
		};

		const auto* mesh_meta = globalMeta->getAssetSpecificMetadata(cpumesh.get());
		auto auxInstanceDataIt = mesh_meta->m_instanceAuxData.begin();
		for (const auto& inst : mesh_meta->m_instances)
		{
			sceneBound.addInternalBox(core::transformBoxEx(cpumesh->getBoundingBox(),inst.worldTform));
			if (auxInstanceDataIt->frontEmitter.type==ext::MitsubaLoader::CElementEmitter::AREA)
			{
				core::vectorSIMDf pos;
				assert(cpumesh->getMeshBuffers().size()==1u);
				const float area = computeAreaAndAvgPos(cpumesh->getMeshBuffers().begin()[0], inst.worldTform, pos);
				assert(area>0.f);
				inst.worldTform.pseudoMulWith4x1(pos);

				SLight l;
				l.intensity = auxInstanceDataIt->frontEmitter.area.radiance*area*2.f*core::PI<float>();
				l.position = pos;

				lights.push_back(l);
			}
			auxInstanceDataIt++;
		}
	}

	constexpr uint32_t ENVMAP_SAMPLE_COUNT = 64u;
	constexpr float LIGHT_INTENSITY_SCALE = 0.01f;

	core::unordered_set<const asset::ICPURenderpassIndependentPipeline*> modifiedPipelines;
	core::unordered_map<core::smart_refctd_ptr<asset::ICPUSpecializedShader>, core::smart_refctd_ptr<asset::ICPUSpecializedShader>> modifiedShaders;
	for (auto& mesh : cpumeshes)
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
				pipeline->getLayout()->setDescriptorSetLayout(2u, core::smart_refctd_ptr(ds2layout));
				auto* fs = pipeline->getShaderAtStage(asset::ICPUSpecializedShader::ESS_FRAGMENT);
				auto found = modifiedShaders.find(core::smart_refctd_ptr<asset::ICPUSpecializedShader>(fs));
				if (found != modifiedShaders.end())
					pipeline->setShaderAtStage(asset::ICPUSpecializedShader::ESS_FRAGMENT, found->second.get());
				else {
					auto newfs = createModifiedFragShader(fs, params.WindowSize.Width, params.WindowSize.Height, lights.size(), ENVMAP_SAMPLE_COUNT, LIGHT_INTENSITY_SCALE);
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


	auto gpumeshes = driver->getGPUObjectsFromAssets(cpumeshes.data(), cpumeshes.data()+cpumeshes.size());

	auto gpuds0 = driver->getGPUObjectsFromAssets(&cpuds0.get(), &cpuds0.get()+1)->front();
    auto gpuds1layout = driver->getGPUObjectsFromAssets(&ds1layout, &ds1layout+1)->front();

    auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(asset::SBasicViewParameters));
    auto gpuds1 = driver->createGPUDescriptorSet(std::move(gpuds1layout));
    {
        video::IGPUDescriptorSet::SWriteDescriptorSet write;
        write.dstSet = gpuds1.get();
        write.binding = ds1UboBinding;
        write.count = 1u;
        write.arrayElement = 0u;
        write.descriptorType = asset::EDT_UNIFORM_BUFFER;
        video::IGPUDescriptorSet::SDescriptorInfo info;
        {
            info.desc = gpuubo;
            info.buffer.offset = 0ull;
            info.buffer.size = gpuubo->getSize();
        }
        write.info = &info;
        driver->updateDescriptorSets(1u, &write, 0u, nullptr);
    }

	smart_refctd_ptr<video::IGPUBufferView> gpuSequenceBufferView;
	{
		constexpr uint32_t MaxSamples = ENVMAP_SAMPLE_COUNT;
		constexpr uint32_t Channels = 3u;

		auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t) * MaxSamples*Channels);

		core::OwenSampler sampler(Channels, 0xdeadbeefu);

		auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
		for (uint32_t c = 0; c < Channels; ++c)
		for (uint32_t i = 0; i < MaxSamples; i++)
		{
			out[Channels*i + c] = sampler.sample(c, i);
		}
		auto gpuSequenceBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sampleSequence->getSize(), sampleSequence->getPointer());
		gpuSequenceBufferView = driver->createGPUBufferView(gpuSequenceBuffer.get(), asset::EF_R32G32B32_UINT);
	}

	smart_refctd_ptr<video::IGPUImageView> gpuScrambleImageView;
	{
		video::IGPUImage::SCreationParams imgParams;
		imgParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
		imgParams.type = asset::IImage::ET_2D;
		imgParams.format = asset::EF_R32G32_UINT;
		imgParams.extent = {params.WindowSize.Width,params.WindowSize.Height,1u};
		imgParams.mipLevels = 1u;
		imgParams.arrayLayers = 1u;
		imgParams.samples = asset::IImage::ESCF_1_BIT;

		video::IGPUImage::SBufferCopy region;
		region.imageExtent = imgParams.extent;
		region.imageSubresource.layerCount = 1u;

		constexpr auto ScrambleStateChannels = 2u;
		const auto renderPixelCount = imgParams.extent.width*imgParams.extent.height;
		core::vector<uint32_t> random(renderPixelCount*ScrambleStateChannels);
		{
			core::RandomSampler rng(0xbadc0ffeu);
			for (auto& pixel : random)
				pixel = rng.nextSample();
		}
		auto buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(random.size()*sizeof(uint32_t),random.data());

		video::IGPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = driver->createFilledDeviceLocalGPUImageOnDedMem(std::move(imgParams),buffer.get(),1u,&region);
		viewParams.viewType = video::IGPUImageView::ET_2D;
		viewParams.format = asset::EF_R32G32_UINT;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.layerCount = 1u;
		gpuScrambleImageView = driver->createGPUImageView(std::move(viewParams));
	}

	auto gpuds2layout = driver->getGPUObjectsFromAssets(&ds2layout.get(), &ds2layout.get()+1)->front();
	auto gpuds2 = driver->createGPUDescriptorSet(std::move(gpuds2layout));
	{
		video::IGPUDescriptorSet::SDescriptorInfo info[4];
		video::IGPUDescriptorSet::SWriteDescriptorSet w[4];
		w[0].arrayElement = 0u;
		w[0].binding = 0u;
		w[0].count = 1u;
		w[0].descriptorType = asset::EDT_STORAGE_BUFFER;
		w[0].dstSet = gpuds2.get();
		w[0].info = info+0;
		auto lightsBuf = driver->createFilledDeviceLocalGPUBufferOnDedMem(lights.size()*sizeof(SLight), lights.data());
		info[0].buffer.offset = 0u;
		info[0].buffer.size = lightsBuf->getSize();
		info[0].desc = std::move(lightsBuf);

		w[1].arrayElement = 0u;
		w[1].binding = 1u;
		w[1].count = 1u;
		w[1].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
		w[1].dstSet = gpuds2.get();
		w[1].info = info+1;
		auto gpuEnvmapImageView = createGPUImageView("../../media/envmap/envmap_0.exr", am, driver);
		info[1].image.imageLayout = asset::EIL_UNDEFINED;
		info[1].image.sampler = nullptr;
		info[1].desc = std::move(gpuEnvmapImageView);

		w[2].arrayElement = 0u;
		w[2].binding = 2u;
		w[2].count = 1u;
		w[2].descriptorType = asset::EDT_UNIFORM_TEXEL_BUFFER;
		w[2].dstSet = gpuds2.get();
		w[2].info = info+2;
		info[2].desc = gpuSequenceBufferView;

		w[3].arrayElement = 0u;
		w[3].binding = 3u;
		w[3].count = 1u;
		w[3].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
		w[3].dstSet = gpuds2.get();
		w[3].info = info+3;
		info[3].image.imageLayout = asset::EIL_UNDEFINED;
		info[3].image.sampler = nullptr;//imm sampler is present
		info[3].desc = gpuScrambleImageView;

		driver->updateDescriptorSets(4u, w, 0u, nullptr);
	}

	// camera and viewport
	scene::ICameraSceneNode* camera = nullptr;
	core::recti viewport(core::position2di(0,0), core::position2di(params.WindowSize.Width,params.WindowSize.Height));

//#define TESTING
#ifdef TESTING
	if (0)
#else
	if (globalMeta->m_global.m_sensors.size() && isOkSensorType(globalMeta->m_global.m_sensors.front()))
#endif
	{
		const auto& film = sensor.film;
		viewport = core::recti(core::position2di(film.cropOffsetX,film.cropOffsetY), core::position2di(film.cropWidth,film.cropHeight));

		auto extent = sceneBound.getExtent();
		camera = smgr->addCameraSceneNodeFPS(nullptr,100.f,core::min(extent.X,extent.Y,extent.Z)*0.0001f);
		// need to extract individual components
		{
			auto relativeTransform = sensor.transform.matrix.extractSub3x4();

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

	constexpr uint32_t MAX_INSTANCES = 512u;
	core::vector<core::matrix4SIMD> uboData(MAX_INSTANCES);
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));
		driver->setViewPort(viewport);
//#ifdef TESTING
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();
//#endif

		asset::SBasicViewParameters uboData;
		//view-projection matrix
		memcpy(uboData.MVP, camera->getConcatenatedMatrix().pointer(), sizeof(core::matrix4SIMD));
		//view matrix
		memcpy(uboData.MV, camera->getViewMatrix().pointer(), sizeof(core::matrix3x4SIMD));
		//writing camera position to 4th column of NormalMat
		uboData.NormalMat[3] = camera->getPosition().X;
		uboData.NormalMat[7] = camera->getPosition().Y;
		uboData.NormalMat[11] = camera->getPosition().Z;
		driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0u, sizeof(uboData), &uboData);

		for (uint32_t j = 0u; j < gpumeshes->size(); ++j)
		{
			auto& mesh = (*gpumeshes)[j];

			for (auto mb : mesh->getMeshBuffers())
			{
				auto* pipeline = mb->getPipeline();
				const video::IGPUDescriptorSet* ds[3]{ gpuds0.get(), gpuds1.get(), gpuds2.get() };
				driver->bindGraphicsPipeline(pipeline);
				driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 0u, 3u, ds, nullptr);
				driver->pushConstants(pipeline->getLayout(), video::IGPUSpecializedShader::ESS_VERTEX|video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, sizeof(float), mb->getPushConstantsDataPtr());

				driver->drawMeshBuffer(mb);
			}
		}

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Mitsuba Loader Demo - Nabla Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

	return 0;
}
