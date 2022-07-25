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

	const asset::SPushConstantRange pc_range = { asset::ISpecializedShader::ESS_COMPUTE, 0u, core::max(sizeof(Parameters_t), sizeof(ScanClass::Parameters_t)) };

	{
		video::IGPUDescriptorSetLayout::SBinding binding = { 0u, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr };
		m_scan_ds_layout = driver->createDescriptorSetLayout(&binding, &binding + 1);
	}

	{
		const uint32_t count = 2u;
		video::IGPUDescriptorSetLayout::SBinding binding[count];
		for (uint32_t i = 0; i < count; ++i)
			binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr };
		m_sort_ds_layout = driver->createDescriptorSetLayout(binding, binding + count);
	}

	m_pipeline_layout = driver->createPipelineLayout(&pc_range, &pc_range + 1, core::smart_refctd_ptr(m_scan_ds_layout),
		core::smart_refctd_ptr(m_sort_ds_layout));

	m_histogram_pipeline = driver->createComputePipeline(nullptr, core::smart_refctd_ptr(m_pipeline_layout),
		createShader("nbl/builtin/glsl/ext/RadixSort/default_histogram.comp", driver));

	m_upsweep_pipeline = driver->createComputePipeline(nullptr, core::smart_refctd_ptr(m_pipeline_layout),
		createShader_Scan("nbl/builtin/glsl/ext/Scan/default_upsweep.comp", driver));

	m_downsweep_pipeline = driver->createComputePipeline(nullptr, core::smart_refctd_ptr(m_pipeline_layout),
		createShader_Scan("nbl/builtin/glsl/ext/Scan/default_downsweep.comp", driver));

	m_scatter_pipeline = driver->createComputePipeline(nullptr, core::smart_refctd_ptr(m_pipeline_layout),
		createShader("nbl/builtin/glsl/ext/RadixSort/default_scatter.comp", driver));
}

void RadixSort::sort(video::IGPUComputePipeline* histogram, video::IGPUComputePipeline* upsweep, video::IGPUComputePipeline* downsweep,
	video::IGPUComputePipeline* scatter, video::IGPUDescriptorSet* ds_scan, core::smart_refctd_ptr<video::IGPUDescriptorSet>* ds_sort,
	ScanClass::Parameters_t* scan_push_constants, Parameters_t* sort_push_constants,
	ScanClass::DispatchInfo_t* scan_dispatch_info, DispatchInfo_t* sort_dispatch_info,
	const uint32_t total_scan_pass_count, const uint32_t upsweep_pass_count,
	video::IVideoDriver* driver)
{
	for (uint32_t pass = 0; pass < PASS_COUNT; ++pass)
	{
		const video::IGPUPipelineLayout* pipeline_layout = histogram->getLayout();
		const video::IGPUDescriptorSet* descriptor_sets[2] = { ds_scan, ds_sort[pass % 2].get() };
		driver->bindDescriptorSets(video::EPBP_COMPUTE, pipeline_layout, 0u, 2u, descriptor_sets, nullptr);

		driver->bindComputePipeline(histogram);
		dispatchHelper(pipeline_layout, sort_push_constants[pass], sort_dispatch_info[0], driver);

		for (uint32_t scan_pass = 0; scan_pass < upsweep_pass_count; ++scan_pass)
		{
			driver->bindComputePipeline(upsweep);
			ScanClass::dispatchHelper(pipeline_layout, scan_push_constants[scan_pass], scan_dispatch_info[scan_pass], driver);
		}

		for (uint32_t scan_pass = upsweep_pass_count; scan_pass < total_scan_pass_count; ++scan_pass)
		{
			driver->bindComputePipeline(downsweep);
			ScanClass::dispatchHelper(pipeline_layout, scan_push_constants[total_scan_pass_count - 1 - scan_pass], scan_dispatch_info[total_scan_pass_count - 1 - scan_pass], driver);
		}

		driver->bindComputePipeline(scatter);
		dispatchHelper(pipeline_layout, sort_push_constants[pass], sort_dispatch_info[0], driver);
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

	auto gpu_shader = driver->createShader(core::smart_refctd_ptr<const asset::ICPUShader>(cpu_specialized_shader->getUnspecialized()));
	auto gpu_shader_specialized = driver->createSpecializedShader(gpu_shader.get(), cpu_specialized_shader->getSpecializationInfo());

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

	auto gpu_shader = driver->createShader(core::smart_refctd_ptr<const asset::ICPUShader>(cpu_specialized_shader->getUnspecialized()));
	auto gpu_shader_specialized = driver->createSpecializedShader(gpu_shader.get(), cpu_specialized_shader->getSpecializationInfo());

	return gpu_shader_specialized;
}

}
}
}