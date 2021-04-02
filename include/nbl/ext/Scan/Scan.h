#ifndef _NBL_EXT_SCAN_INCLUDED_

#include "nabla.h"

namespace nbl
{
namespace ext
{
namespace Scan
{

typedef uint32_t uint;
#include "nbl/builtin/glsl/ext/Scan/parameters_struct.glsl"

class Scan final : public core::IReferenceCounted
{
public:
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORKGROUP_SIZE = 256u;

	enum Operator : uint32_t
	{
		AND = 1 << 0,
		XOR = 1 << 1,
		OR  = 1 << 2,
		ADD = 1 << 3,
		MUL = 1 << 4,
		MIN = 1 << 5,
		MAX = 1 << 6,
	};

	typedef nbl_glsl_ext_Scan_Parameters_t Parameters_t;

	struct DispatchInfo_t
	{
		uint32_t upsweep_pass_count;
		uint32_t downsweep_pass_count;
		uint32_t wg_count;
		std::stack<uint32_t> element_count_pass_stack;
	};

	Scan(video::IDriver* driver, Operator op, const uint32_t wg_size);

	inline auto getDefaultDescriptorSetLayout() const { return m_ds_layout.get(); }

	inline auto getDefaultPipelineLayout() const { return m_pipeline_layout.get(); }

	inline auto getDefaultUpsweepPipeline() const { return m_upsweep_pipeline.get(); }

	inline auto getDefaultDownsweepPipeline() const { return m_downsweep_pipeline.get(); }

	static inline void dispatchHelper(const video::IGPUPipelineLayout* pipeline_layout, const nbl_glsl_ext_Scan_Parameters_t& params,
		const DispatchInfo_t& dispatch_info, video::IVideoDriver* driver, bool issue_default_barrier = true)
	{
		driver->pushConstants(pipeline_layout, asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(Parameters_t), &params);
		driver->dispatch(dispatch_info.wg_count, 1, 1);

		if (issue_default_barrier)
			defaultBarrier();
	}

	static inline void buildParameters(const uint32_t in_count, Parameters_t* push_constants, DispatchInfo_t* dispatch_info,
		const uint32_t wg_size)
	{
		dispatch_info->upsweep_pass_count = std::ceil(log(in_count) / log(wg_size));
		assert(dispatch_info->upsweep_pass_count != 0u && "Input element count should be > 1!");

		dispatch_info->downsweep_pass_count = dispatch_info->upsweep_pass_count - 1u;

		push_constants->element_count_pass = in_count;
		push_constants->element_count_total = in_count;
		push_constants->stride = 1u;
	}

	static inline void prePassParameterUpdate(uint32_t pass_idx, bool is_upsweep, Parameters_t* push_constants,
		DispatchInfo_t* dispatch_info, const uint32_t wg_size)
	{
		if (is_upsweep)
		{
			if (pass_idx != (dispatch_info->upsweep_pass_count - 1u))
				dispatch_info->element_count_pass_stack.push(push_constants->element_count_pass);
		}
		else
		{
			push_constants->element_count_pass = dispatch_info->element_count_pass_stack.top();
			dispatch_info->element_count_pass_stack.pop();
		}

		dispatch_info->wg_count = (push_constants->element_count_pass + wg_size - 1) / wg_size;
	}

	static inline void postPassParameterUpdate(uint32_t pass_idx, bool is_upsweep, Parameters_t* push_constants, DispatchInfo_t* dispatch_info,
		const uint32_t wg_size)
	{
		if (is_upsweep && (pass_idx != (dispatch_info->upsweep_pass_count - 1u)))
		{
			push_constants->stride *= wg_size;
			push_constants->element_count_pass = dispatch_info->wg_count;
		}
		else
		{
			push_constants->stride /= wg_size;
		}
	}

	static inline void defaultBarrier()
	{
		video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	static void updateDescriptorSet(video::IGPUDescriptorSet* set, core::smart_refctd_ptr<video::IGPUBuffer> descriptor, video::IVideoDriver* driver);

	const uint32_t m_wg_size;

private:
	~Scan() {}

	core::smart_refctd_ptr<video::IGPUPipelineLayout> m_pipeline_layout = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_ds_layout = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_upsweep_pipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_downsweep_pipeline = nullptr;

	core::smart_refctd_ptr<video::IGPUComputePipeline> createPipeline(const char* shader_include_name, Operator bin_op, video::IDriver* driver);
};

}
}
}

#define _NBL_EXT_SCAN_INCLUDED_
#endif