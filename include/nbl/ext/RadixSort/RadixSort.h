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
	static inline const uint32_t DEFAULT_WORKGROUP_SIZE = 256u;
	static inline const uint32_t BITS_PER_PASS = 4u;
	static inline const uint32_t PASS_COUNT = 32u / BITS_PER_PASS;
	static inline const uint32_t BUCKETS_COUNT = 1 << BITS_PER_PASS;

	typedef nbl_glsl_ext_RadixSort_Parameters_t Parameters_t;

	struct DispatchInfo_t
	{
		uint32_t wg_count[3];
	};

	const uint32_t m_wg_size;
	core::smart_refctd_ptr<ScanClass> m_scanner = nullptr;
	
	RadixSort(video::IDriver* driver, const uint32_t wg_size);

	static void sort(video::IGPUComputePipeline* histogram, video::IGPUComputePipeline* upsweep, video::IGPUComputePipeline* downsweep,
		video::IGPUComputePipeline* scatter, video::IGPUDescriptorSet* ds_scan, core::smart_refctd_ptr<video::IGPUDescriptorSet>* ds_sort,
		ScanClass::Parameters_t* scan_push_constants, Parameters_t* sort_push_constants,
		ScanClass::DispatchInfo_t* scan_dispatch_info, DispatchInfo_t* sort_dispatch_info,
		const uint32_t total_scan_pass_count, const uint32_t upsweep_pass_count,
		video::IVideoDriver* driver);

	inline auto getDefaultScanDescriptorSetLayout() const { return m_scan_ds_layout.get(); }
	inline auto getDefaultSortDescriptorSetLayout() const { return m_sort_ds_layout.get(); }

	inline auto getDefaultPipelineLayout() const { return m_pipeline_layout.get(); }

	inline auto getDefaultHistogramPipeline() const { return m_histogram_pipeline.get(); }
	inline auto getDefaultUpsweepPipeline() const { return m_upsweep_pipeline.get(); }
	inline auto getDefaultDownsweepPipeline() const { return m_downsweep_pipeline.get(); }
	inline auto getDefaultScatterPipeline() const { return m_scatter_pipeline.get(); }

	static inline void dispatchHelper(const video::IGPUPipelineLayout* pipeline_layout, const nbl_glsl_ext_RadixSort_Parameters_t& params,
		const DispatchInfo_t& dispatch_info, video::IVideoDriver* driver, bool issue_default_barrier = true)
	{
		// Since we're using a single pc range we need to update this for both the radix sort exclusive pipelines (histogram and scatter)
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
		{
			sort_push_constants[pass].shift = BITS_PER_PASS * pass;
			sort_push_constants[pass].element_count_total = in_count;
		}

		return total_scan_pass_count;
	}

	static inline void updateDescriptorSetsPingPong(core::smart_refctd_ptr<video::IGPUDescriptorSet>* pingpong_sets, const asset::SBufferRange<video::IGPUBuffer>& range_zero,
		const asset::SBufferRange<video::IGPUBuffer>& range_one, video::IVideoDriver* driver)
	{
		const uint32_t count = 2u;
		asset::SBufferRange<video::IGPUBuffer> ranges[count];
		for (uint32_t i = 0; i < 2u; ++i)
		{
			if (i == 0)
			{
				ranges[0] = range_zero;
				ranges[1] = range_one;
			}
			else
			{
				ranges[0] = range_one;
				ranges[1] = range_zero;
			}
			updateDescriptorSet(pingpong_sets[i].get(), ranges, count, driver);
		}
	}

	static inline void updateDescriptorSet(video::IGPUDescriptorSet* ds, const asset::SBufferRange<video::IGPUBuffer>* descriptor_ranges,
		const uint32_t count, video::IDriver* driver)
	{
		constexpr uint32_t MAX_DESCRIPTOR_COUNT = 2u;
		assert(count <= MAX_DESCRIPTOR_COUNT);

		video::IGPUDescriptorSet::SDescriptorInfo ds_info[MAX_DESCRIPTOR_COUNT];
		video::IGPUDescriptorSet::SWriteDescriptorSet writes[MAX_DESCRIPTOR_COUNT];

		for (uint32_t i = 0; i < count; ++i)
		{
			ds_info[i].desc = descriptor_ranges[i].buffer;
			ds_info[i].buffer = { descriptor_ranges[i].offset, descriptor_ranges[i].size };

			writes[i] = { ds, i, 0u, 1u, asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER, ds_info + i };
		}

		driver->updateDescriptorSets(count, writes, 0u, nullptr);
	}

	static inline void defaultBarrier()
	{
		video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

private:
	~RadixSort() {}

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_scan_ds_layout = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_sort_ds_layout = nullptr;

	core::smart_refctd_ptr<video::IGPUPipelineLayout> m_pipeline_layout = nullptr;

	core::smart_refctd_ptr<video::IGPUComputePipeline> m_histogram_pipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_upsweep_pipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_downsweep_pipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_scatter_pipeline = nullptr;

	core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader(const char* shader_file_path, video::IDriver* driver);
	core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader_Scan(const char* shader_file_path, video::IDriver* driver);
};

}
}
}

#define _NBL_EXT_RADIXSORT_INCLUDED_
#endif