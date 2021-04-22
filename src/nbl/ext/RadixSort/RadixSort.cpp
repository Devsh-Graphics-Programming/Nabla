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

	{
		const asset::SPushConstantRange pc_range = { asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(ScanClass::Parameters_t) };
		video::IGPUDescriptorSetLayout::SBinding binding = { 0u, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr };

		m_scan_ds_layout = driver->createGPUDescriptorSetLayout(&binding, &binding + 1);
		m_scan_pipeline_layout = driver->createGPUPipelineLayout(&pc_range, &pc_range + 1, core::smart_refctd_ptr(m_scan_ds_layout));

		m_upsweep_pipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_scan_pipeline_layout),
			createShader_Scan("nbl/builtin/glsl/ext/Scan/default_upsweep.comp", driver));

		m_downsweep_pipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_scan_pipeline_layout),
			createShader_Scan("nbl/builtin/glsl/ext/Scan/default_downsweep.comp", driver));
	}

	const asset::SPushConstantRange pc_range = { asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(nbl_glsl_ext_RadixSort_Parameters_t) };

	{
		// Todo: I need this to be made 2u
		const uint32_t count = 3u;
		video::IGPUDescriptorSetLayout::SBinding binding[count];
		for (uint32_t i = 0; i < count; ++i)
			binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr };

		m_sort_ds_layout = driver->createGPUDescriptorSetLayout(binding, binding + count);
		m_sort_pipeline_layout = driver->createGPUPipelineLayout(&pc_range, &pc_range + 1, core::smart_refctd_ptr(m_sort_ds_layout));

		m_histogram_pipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_sort_pipeline_layout),
			createShader("nbl/builtin/glsl/ext/RadixSort/default_histogram.comp", driver));

		m_scatter_pipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_sort_pipeline_layout),
			createShader("nbl/builtin/glsl/ext/RadixSort/default_scatter.comp", driver));
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

core::smart_refctd_ptr<video::IGPUSpecializedShader> RadixSort::createShader_Scan(const char* shader_file_path, video::IDriver* driver)
{
	const char* source_fmt =
R"===(#version 430 core

#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_SCAN_BIN_OP_ (1 << 3)
#define _NBL_GLSL_EXT_SCAN_STORAGE_TYPE_ uint

layout (local_size_x = _NBL_GLSL_WORKGROUP_SIZE_) in;
 
#include "%s"

)===";

	// Todo: This just the value I took from FFT example, don't know how it is being computed.
	const size_t extraSize = 4u + 8u + 8u + 128u;

	auto shader = core::make_smart_refctd_ptr<asset::ICPUBuffer>(strlen(source_fmt) + extraSize + 1u);
	snprintf(reinterpret_cast<char*>(shader->getPointer()), shader->getSize(), source_fmt, m_wg_size, shader_file_path);

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