#include "nbl/ext/Scan/Scan.h"

using namespace nbl::ext::Scan;

Scan::Scan(video::IDriver* driver, Operator op, const uint32_t wg_size) : m_wg_size(wg_size)
{
	assert(nbl::core::isPoT(wg_size));

	const asset::SPushConstantRange pc_range = { asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(nbl_glsl_ext_Scan_Parameters_t) };
	video::IGPUDescriptorSetLayout::SBinding binding = { 0u, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr };

	m_ds_layout = driver->createGPUDescriptorSetLayout(&binding, &binding + 1);
	m_pipeline_layout = driver->createGPUPipelineLayout(&pc_range, &pc_range + 1, core::smart_refctd_ptr(m_ds_layout));

	m_upsweep_pipeline = createPipeline("nbl/builtin/glsl/ext/Scan/default_upsweep.comp", op, driver);
	m_downsweep_pipeline = createPipeline("nbl/builtin/glsl/ext/Scan/default_downsweep.comp", op, driver);
}

void Scan::updateDescriptorSet(video::IGPUDescriptorSet* set, core::smart_refctd_ptr<video::IGPUBuffer> descriptor,
	video::IVideoDriver* driver)
{
	video::IGPUDescriptorSet::SDescriptorInfo ds_info = {};
	ds_info.desc = descriptor;
	ds_info.buffer = { 0u, descriptor->getSize() };

	video::IGPUDescriptorSet::SWriteDescriptorSet writes = { set, 0, 0u, 1u, asset::EDT_STORAGE_BUFFER, &ds_info };

	driver->updateDescriptorSets(1, &writes, 0u, nullptr);
}

nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline> Scan::createPipeline(const char* shader_include_name, Operator bin_op, video::IDriver* driver)
{
	const char* source_fmt =
R"===(#version 430 core

#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_SCAN_BIN_OP_ %u

layout (local_size_x = _NBL_GLSL_WORKGROUP_SIZE_) in;
 
#include "%s"

)===";

	// Todo: This just the value I took from FFT example, don't know it is being computed.
	const size_t extraSize = 4u + 8u + 8u + 128u;

	auto shader = core::make_smart_refctd_ptr<asset::ICPUBuffer>(strlen(source_fmt) + extraSize + 1u);
	snprintf(reinterpret_cast<char*>(shader->getPointer()), shader->getSize(), source_fmt, m_wg_size, bin_op, shader_include_name);

	auto cpu_specialized_shader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shader), asset::ICPUShader::buffer_contains_glsl),
		asset::ISpecializedShader::SInfo{ nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE });

	auto gpu_shader = driver->createGPUShader(core::smart_refctd_ptr<const asset::ICPUShader>(cpu_specialized_shader->getUnspecialized()));
	auto gpu_shader_specialized = driver->createGPUSpecializedShader(gpu_shader.get(), cpu_specialized_shader->getSpecializationInfo());

	return driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_pipeline_layout), std::move(gpu_shader_specialized));
}