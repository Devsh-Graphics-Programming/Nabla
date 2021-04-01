#ifndef _NBL_EXT_SCAN_INCLUDED_

#include "nabla.h"

typedef uint32_t uint;
#include "nbl/builtin/glsl/ext/Scan/parameters_struct.glsl"

namespace nbl
{
namespace ext
{
namespace Scan
{

class Scan final : public core::IReferenceCounted
{
public:
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORKGROUP_SIZE = 256u;

	enum Operator : uint32_t
	{
		AND = 1 << 0,
		XOR = 1 << 1,
		OR = 1 << 2,
		ADD = 1 << 3,
		MUL = 1 << 4,
		MIN = 1 << 5,
		MAX = 1 << 6,
	};

    Scan(video::IDriver* driver, Operator op)
    {
		const asset::SPushConstantRange pc_range = { asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(nbl_glsl_ext_Scan_Parameters_t) };
		video::IGPUDescriptorSetLayout::SBinding binding = { 0u, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr };

		m_ds_layout = driver->createGPUDescriptorSetLayout(&binding, &binding + 1);
		m_pipeline_layout = driver->createGPUPipelineLayout(&pc_range, &pc_range + 1, core::smart_refctd_ptr(m_ds_layout));
		
		m_upsweep_pipeline = createPipeline("nbl/builtin/glsl/ext/Scan/default_upsweep.comp", op, driver);
		m_downsweep_pipeline = createPipeline("nbl/builtin/glsl/ext/Scan/default_downsweep.comp", op, driver);
    }

	inline auto getDefaultDescriptorSetLayout() const { return m_ds_layout.get(); }

	inline auto getDefaultPipelineLayout() const { return m_pipeline_layout.get(); }

	inline auto getDefaultUpsweepPipeline() const { return m_upsweep_pipeline.get(); }

	inline auto getDefaultDownsweepPipeline() const { return m_downsweep_pipeline.get(); }

	static void updateDescriptorSet(video::IGPUDescriptorSet* set, core::smart_refctd_ptr<video::IGPUBuffer> descriptor,
		video::IVideoDriver* driver)
	{
		video::IGPUDescriptorSet::SDescriptorInfo ds_info = {};
		ds_info.desc = descriptor;
		ds_info.buffer = { 0u, descriptor->getSize() };

		video::IGPUDescriptorSet::SWriteDescriptorSet writes = { set, 0, 0u, 1u, asset::EDT_STORAGE_BUFFER, &ds_info };

		driver->updateDescriptorSets(1, &writes, 0u, nullptr);
	}

private:
	~Scan() {}

	core::smart_refctd_ptr<video::IGPUPipelineLayout> m_pipeline_layout = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_ds_layout = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_upsweep_pipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_downsweep_pipeline = nullptr;

	core::smart_refctd_ptr<video::IGPUComputePipeline> createPipeline(const char* shader_include_name, Operator bin_op, video::IDriver* driver)
	{
		const char* source_fmt =
R"===(#version 430 core

#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_SCAN_BIN_OP_ %u

layout (local_size_x = _NBL_GLSL_WORKGROUP_SIZE_) in;
 
#include "%s"

)===";

		// Question: How is this being computed? This just the value I took from FFT example.
		const size_t extraSize = 4u + 8u + 8u + 128u;

		auto shader = core::make_smart_refctd_ptr<asset::ICPUBuffer>(strlen(source_fmt) + extraSize + 1u);
		snprintf(reinterpret_cast<char*>(shader->getPointer()), shader->getSize(), source_fmt, DEFAULT_WORKGROUP_SIZE, bin_op, shader_include_name);

		auto cpu_specialized_shader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(
			core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shader), asset::ICPUShader::buffer_contains_glsl),
			asset::ISpecializedShader::SInfo{ nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE });

		auto gpu_shader = driver->createGPUShader(core::smart_refctd_ptr<const asset::ICPUShader>(cpu_specialized_shader->getUnspecialized()));
		auto gpu_shader_specialized = driver->createGPUSpecializedShader(gpu_shader.get(), cpu_specialized_shader->getSpecializationInfo());

		return driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_pipeline_layout), std::move(gpu_shader_specialized));
	}
};

}
}
}

#define _NBL_EXT_SCAN_INCLUDED_
#endif