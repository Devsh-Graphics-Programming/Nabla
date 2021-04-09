#include "nbl/ext/RadixSort/RadixSort.h"

namespace nbl
{
namespace ext
{
namespace RadixSort
{

RadixSort::RadixSort(video::IDriver* driver, const uint32_t wg_size) : m_wg_size(wg_size)
{
	assert(nbl::core::isPoT(m_wg_size));

	const asset::SPushConstantRange pc_range = { asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(nbl_glsl_ext_RadixSort_Parameters_t) };

	{
		const uint32_t count = 2u;
		video::IGPUDescriptorSetLayout::SBinding binding[count];
		for (uint32_t i = 0; i < count; ++i)
			binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr };

		m_histogram_ds_layout = driver->createGPUDescriptorSetLayout(binding, binding + count);
		m_histogram_pipeline_layout = driver->createGPUPipelineLayout(&pc_range, &pc_range + 1, core::smart_refctd_ptr(m_histogram_ds_layout));

		m_histogram_pipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_histogram_pipeline_layout),
			createShader("nbl/builtin/glsl/ext/RadixSort/default_histogram.comp", driver));
	}

	{
		const uint32_t count = 3u;
		video::IGPUDescriptorSetLayout::SBinding binding[count];
		for (uint32_t i = 0; i < count; ++i)
			binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr };

		m_scatter_ds_layout = driver->createGPUDescriptorSetLayout(binding, binding + count);
		m_scatter_pipeline_layout = driver->createGPUPipelineLayout(&pc_range, &pc_range + 1, core::smart_refctd_ptr(m_scatter_ds_layout));

		m_scatter_pipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_scatter_pipeline_layout),
			createShader("nbl/builtin/glsl/ext/RadixSort/default_scatter.comp", driver));
	}

	m_scanner = core::make_smart_refctd_ptr<ScanClass>(driver, ScanClass::Operator::ADD, m_wg_size);
}

void RadixSort::exclusiveSumScan(video::IVideoDriver* driver, core::smart_refctd_ptr<video::IGPUBuffer> in_gpu,
	video::IGPUDescriptorSet* ds_upsweep, video::IGPUComputePipeline* upsweep_pipeline, video::IGPUDescriptorSet* ds_downsweep,
	video::IGPUComputePipeline* downsweep_pipeline, ScanClass::Parameters_t* push_constants, ScanClass::DispatchInfo_t* dispatch_info,
	const uint32_t total_pass_count, const uint32_t upsweep_pass_count)
{
	for (uint32_t pass = 0; pass < total_pass_count; ++pass)
	{
		if (pass < upsweep_pass_count)
		{
			ScanClass::updateDescriptorSet(ds_upsweep, in_gpu, driver);

			driver->bindComputePipeline(upsweep_pipeline);
			driver->bindDescriptorSets(video::EPBP_COMPUTE, upsweep_pipeline->getLayout(), 0u, 1u, &ds_upsweep, nullptr);
			ScanClass::dispatchHelper(upsweep_pipeline->getLayout(), push_constants[pass], dispatch_info[pass], driver);
		}
		else
		{
			ScanClass::updateDescriptorSet(ds_downsweep, in_gpu, driver);

			driver->bindComputePipeline(downsweep_pipeline);
			driver->bindDescriptorSets(video::EPBP_COMPUTE, downsweep_pipeline->getLayout(), 0u, 1u, &ds_downsweep, nullptr);
			ScanClass::dispatchHelper(downsweep_pipeline->getLayout(), push_constants[total_pass_count - 1 - pass], dispatch_info[total_pass_count - 1 - pass], driver);
		}
	}
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> RadixSort::createShader(const char* shader_file_path, video::IDriver* driver)
{
	const char* source_fmt =
R"===(#version 450

#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_ %u

layout (local_size_x = _NBL_GLSL_WORKGROUP_SIZE_) in;
 
#include "%s"

)===";

	// Todo: This just the value I took from FFT example, don't know how it is being computed.
	const size_t extraSize = 4u + 8u + 8u + 128u;

	auto shader = core::make_smart_refctd_ptr<asset::ICPUBuffer>(strlen(source_fmt) + extraSize + 1u);
	snprintf(reinterpret_cast<char*>(shader->getPointer()), shader->getSize(), source_fmt, m_wg_size, BUCKETS_COUNT, shader_file_path);

	auto cpu_specialized_shader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shader), asset::ICPUShader::buffer_contains_glsl),
		asset::ISpecializedShader::SInfo{ nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE });

	auto gpu_shader = driver->createGPUShader(core::smart_refctd_ptr<const asset::ICPUShader>(cpu_specialized_shader->getUnspecialized()));
	auto gpu_shader_specialized = driver->createGPUSpecializedShader(gpu_shader.get(), cpu_specialized_shader->getSpecializationInfo());

	return gpu_shader_specialized;
}

}
}
}