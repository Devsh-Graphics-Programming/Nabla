#ifndef _NBL_EXT_RADIXSORT_INCLUDED_

#include "nabla.h"
#include "nbl/ext/Scan/Scan.h"

namespace nbl
{
namespace ext
{
namespace RadixSort
{

using ScanClass = ext::Scan::Scan<uint32_t>;

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
		uint32_t wg_count[3];
	};

	const uint32_t m_wg_size;
	core::smart_refctd_ptr<ScanClass> m_scanner = nullptr;
	
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
		driver->dispatch(dispatch_info.wg_count[0], 1, 1);

		if (issue_default_barrier)
			defaultBarrier();
	}

	// Returns the total number of passes required by scan, since the total number of passes required by the radix sort
	// is always constant and is given by `PASS_COUNT` constant above
	static inline uint32_t buildParameters(const uint32_t in_count, const uint32_t wg_size, Parameters_t* sort_push_constants,
		DispatchInfo_t* sort_dispatch_info, ScanClass::Parameters_t* scan_push_constants, ScanClass::DispatchInfo_t* scan_dispatch_info)
	{
		const uint32_t wg_count = (in_count + wg_size - 1) / wg_size;
		const uint32_t histogram_count = wg_count * BUCKETS_COUNT;

		const uint32_t total_scan_pass_count = ScanClass::buildParameters(histogram_count, wg_size, scan_push_constants, scan_dispatch_info);

		if (!scan_push_constants || !scan_dispatch_info || !sort_push_constants || !sort_dispatch_info)
			return total_scan_pass_count;

		sort_dispatch_info[0] = { {wg_count, 0, 0} };

		for (uint32_t pass = 0; pass < PASS_COUNT; ++pass)
			sort_push_constants[pass] = { BITS_PER_PASS * pass,  in_count };

		return total_scan_pass_count;
	}

	static inline void defaultBarrier()
	{
		video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	// Todo: Don't let it use core::vector
	static inline void updateDescriptorSet(video::IGPUDescriptorSet* set, core::vector< asset::SBufferRange<video::IGPUBuffer> > descriptors,
		video::IVideoDriver* driver)
	{
		constexpr uint32_t MAX_DESCRIPTOR_COUNT = 3u;
		assert(descriptors.size() <= MAX_DESCRIPTOR_COUNT);

		video::IGPUDescriptorSet::SDescriptorInfo ds_info[MAX_DESCRIPTOR_COUNT];
		for (uint32_t i = 0; i < descriptors.size(); ++i)
		{
			ds_info[i].desc = descriptors[i].buffer;
			ds_info[i].buffer = { descriptors[i].offset, descriptors[i].size };
		}

		video::IGPUDescriptorSet::SWriteDescriptorSet writes[MAX_DESCRIPTOR_COUNT];
		for (uint32_t i = 0; i < descriptors.size(); ++i)
			writes[i] = { set, i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };

		driver->updateDescriptorSets(descriptors.size(), writes, 0u, nullptr);
	}

	static void exclusiveSumScan(video::IVideoDriver* driver, core::smart_refctd_ptr<video::IGPUBuffer> in_gpu,
		video::IGPUDescriptorSet* ds_upsweep, video::IGPUComputePipeline* upsweep_pipeline, video::IGPUDescriptorSet* ds_downsweep,
		video::IGPUComputePipeline* downsweep_pipeline, ScanClass::Parameters_t* push_constants, ScanClass::DispatchInfo_t* dispatch_info,
		const uint32_t total_pass_count, const uint32_t upsweep_pass_count);

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