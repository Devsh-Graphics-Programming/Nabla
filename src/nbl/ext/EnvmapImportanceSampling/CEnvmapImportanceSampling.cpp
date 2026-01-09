#include "nbl/ext/EnvmapImportanceSampling/CEnvmapImportanceSampling.h"
#include "nbl/ext/EnvmapImportanceSampling/builtin/hlsl/common.hlsl"

using namespace nbl::hlsl::ext::envmap_importance_sampling;

#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/ext/debug_draw/builtin/build/CArchive.h"
#endif

#include "nbl/ext/EnvmapImportanceSampling/builtin/build/spirv/keys.hpp"

using namespace nbl;
using namespace core;
using namespace video;
using namespace system;
using namespace asset;
using namespace hlsl;

namespace nbl::ext::envmap_importance_sampling
{

constexpr std::string_view NBL_EXT_MOUNT_ENTRY = "nbl/ext/EnvmapImportanceSampling";

const smart_refctd_ptr<IFileArchive> EnvmapImportanceSampling::mount(core::smart_refctd_ptr<ILogger> logger, ISystem* system, video::ILogicalDevice* device, const std::string_view archiveAlias)
{
  assert(system);

	if (!system)
		return nullptr;

	// extension should mount everything for you, regardless if content goes from virtual filesystem 
	// or disk directly - and you should never rely on application framework to expose extension data
	#ifdef NBL_EMBED_BUILTIN_RESOURCES
	auto archive = make_smart_refctd_ptr<builtin::build::CArchive>(smart_refctd_ptr(logger));
	#else
	auto archive = make_smart_refctd_ptr<nbl::system::CMountDirectoryArchive>(std::string_view(NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT), smart_refctd_ptr(logger), system);
	#endif

	system->mount(smart_refctd_ptr(archive), archiveAlias.data());
	return smart_refctd_ptr(archive);
}

core::smart_refctd_ptr<video::IGPUComputePipeline> EnvmapImportanceSampling::createGenLumaPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout)
{
	system::logger_opt_ptr logger = params.utilities->getLogger();
	auto system = smart_refctd_ptr<ISystem>(params.assetManager->getSystem());
	auto* device = params.utilities->getLogicalDevice();
  mount(smart_refctd_ptr<ILogger>(params.utilities->getLogger()), system.get(), params.utilities->getLogicalDevice(), NBL_EXT_MOUNT_ENTRY);

	auto getShader = [&](const core::string& key)->smart_refctd_ptr<IShader> {
		IAssetLoader::SAssetLoadParams lp = {};
		lp.logger = params.utilities->getLogger();
		lp.workingDirectory = NBL_EXT_MOUNT_ENTRY;
		auto bundle = params.assetManager->getAsset(key.c_str(), lp);

		const auto contents = bundle.getContents();

		if (contents.empty())
		{
			logger.log("Failed to load shader %s from disk", ILogger::ELL_ERROR, key.c_str());
			return nullptr;
		}

		if (bundle.getAssetType() != IAsset::ET_SHADER)
		{
			logger.log("Loaded asset has wrong type!", ILogger::ELL_ERROR);
			return nullptr;
		}

		return IAsset::castDown<IShader>(contents[0]);
	};

	const auto key = nbl::ext::envmap_importance_sampling::builtin::build::get_spirv_key<"measure_luma">(device);
	smart_refctd_ptr<IShader> genLumaShader = getShader(key);
	if (!genLumaShader)
	{
		params.utilities->getLogger()->log("Could not compile shaders!", ILogger::ELL_ERROR);
		return nullptr;
	}

	return nullptr;

}

//
// core::smart_refctd_ptr < video::IGPUPipelineLayout> EnvmapImportanceSampling::createLumaGenPipelineLayout(video::ILogicalDevice* device)
// {
//   asset::SPushConstantRange pcRange = {
//     .stageFlags = hlsl::ESS_COMPUTE,
//     .offset = 0,
//     .size = sizeof(SLumaGenPushConstants)
//   };
//
//   const IGPUDescriptorSetLayout::SBinding bindings[] = {
//     {
//       .binding = 0u,
//       .type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
//       .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
//       .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
//       .count = 1u,
//       .immutableSamplers = &defaultSampler
//     },
//     {
//       .binding = 1u,
//       .type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
//       .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
//       .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
//       .count = 1u
//     }
//   };
//
// }

}
