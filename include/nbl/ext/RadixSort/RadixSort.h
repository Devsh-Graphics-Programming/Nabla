#ifndef _NBL_EXT_RADIXSORT_INCLUDED_

#include "nabla.h"

namespace nbl
{
namespace ext
{
namespace RadixSort
{

typedef uint32_t uint;
#include "nbl/builtin/glsl/ext/RadixSort/parameters_struct.glsl"

class RadixSort final : public core::IReferenceCounted
{
public:
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORKGROUP_SIZE = 256u;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t BITS_PER_PASS = 4u;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t PASS_COUNT = 32u / BITS_PER_PASS;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t BUCKETS_COUNT = 1 << BITS_PER_PASS;

	typedef nbl_glsl_ext_RadixSort_Parameters_t Parameters_t;

	struct DispatchInfo_t
	{
		uint32_t wg_count;
		uint32_t histogram_count;
	};
	
	RadixSort(video::IDriver* driver, const uint32_t wg_size);

	inline auto getDefaultHistogramDescriptorSetLayout() const { return m_histogram_ds_layout.get(); }
	inline auto getDefaultHistogramPipelineLayout() const { return m_histogram_pipeline_layout.get(); }
	inline auto getDefaultHistogramPipeline() const { return m_histogram_pipeline.get(); }

	inline auto getDefaultScatterDescriptorSetLayout() const { return m_scatter_ds_layout.get(); }
	inline auto getDefaultScatterPipelineLayout() const { return m_scatter_pipeline_layout.get(); }
	inline auto getDefaultScatterPipeline() const { return m_scatter_pipeline.get(); }

	static inline void dispatchHelper(const video::IGPUPipelineLayout* pipeline_layout, const nbl_glsl_ext_RadixSort_Parameters_t& params,
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
		push_constants->element_count_total = in_count;
		push_constants->shift = 0u;

		dispatch_info->wg_count = (in_count + wg_size - 1) / wg_size;
		dispatch_info->histogram_count = dispatch_info->wg_count * BUCKETS_COUNT;
	}

	static inline void updateParameters(Parameters_t* push_constants, const uint32_t pass_idx)
	{
		push_constants->shift = 4u * pass_idx;
	}

	static inline void defaultBarrier()
	{
		video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	static inline void updateDescriptorSet(video::IGPUDescriptorSet* set, core::vector< core::smart_refctd_ptr<video::IGPUBuffer> > descriptors,
		video::IVideoDriver* driver)
	{
		constexpr uint32_t MAX_DESCRIPTOR_COUNT = 3u;
		assert(descriptors.size() <= MAX_DESCRIPTOR_COUNT);

		video::IGPUDescriptorSet::SDescriptorInfo ds_info[MAX_DESCRIPTOR_COUNT];
		for (uint32_t i = 0; i < descriptors.size(); ++i)
		{
			ds_info[i].desc = descriptors[i];
			ds_info[i].buffer = { 0u, descriptors[i]->getSize() };
		}

		video::IGPUDescriptorSet::SWriteDescriptorSet writes[MAX_DESCRIPTOR_COUNT];
		for (uint32_t i = 0; i < descriptors.size(); ++i)
			writes[i] = { set, i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };

		driver->updateDescriptorSets(descriptors.size(), writes, 0u, nullptr);
	}

	const uint32_t m_wg_size;

private:
	~RadixSort() {}

	core::smart_refctd_ptr<video::IGPUPipelineLayout> m_histogram_pipeline_layout = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_histogram_ds_layout = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_histogram_pipeline = nullptr;

	core::smart_refctd_ptr<video::IGPUPipelineLayout> m_scatter_pipeline_layout = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_scatter_ds_layout = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_scatter_pipeline = nullptr;

	core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader(const char* shader_file_path, video::IDriver* driver);
};

}
}
}

#define _NBL_EXT_RADIXSORT_INCLUDED_
#endif