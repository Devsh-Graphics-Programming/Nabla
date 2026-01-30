#include "nbl/builtin/hlsl/sampling/warps/spherical.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic.hlsl"

#include "common.hlsl"

using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::hlsl::ext::envmap_importance_sampling;

// TODO(kevinyu): Temporary to make nsc works
using config_t = WORKGROUP_CONFIG_T;

[[vk::push_constant]] SLumaMeasurePushConstants pc;

[[vk::binding(0, 0)]] Texture2D<float32_t> lumaMap;

// final (level 1/2) scan needs to fit in one subgroup exactly
groupshared float32_t scratch[mpl::max_v<int16_t,config_t::SharedScratchElementCount,1>];

struct PreloadedUnitData
{
    float32_t3 weightedDir;
    float32_t luma;
};

struct ScratchProxy
{
    template<typename AccessType, typename IndexType>
    void get(const uint32_t ix, NBL_REF_ARG(AccessType) value)
    {
      value = scratch[ix];
    }

    template <typename AccessType, typename IndexType>
    void set(const uint32_t ix, const AccessType value)
    {
        scratch[ix] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
    }
};

struct PreloadedData
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(1u) << config_t::WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t PreloadedDataCount = config_t::VirtualWorkgroupSize / WorkgroupSize;

    PreloadedUnitData getData(const uint32_t ix)
    {
      PreloadedUnitData value;
      const int32_t2 pixelCoord = int32_t2(ix % pc.lumaMapResolution.x, ix / pc.lumaMapResolution.x);
      const float32_t2 uv = (float32_t2(pixelCoord) + float32_t2(0.5, 0.5)) / float32_t2(pc.lumaMapResolution);
      const float32_t luma = lumaMap.Load(int32_t3(pixelCoord, 0));
      value.weightedDir = sampling::warp::Spherical::warp(uv).dst * luma;
      value.luma = luma;
      return value;
    }

    void preload()
    {
        const uint16_t invocationIndex = hlsl::workgroup::SubgroupContiguousIndex();
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data[idx] = getData(idx * WorkgroupSize + invocationIndex);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
    }

    PreloadedUnitData data[config_t::ItemsPerInvocation_0];
};

static PreloadedData preloadData;

struct DirXAccessor
{
    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
      value = preloadData.data[ix >> config_t::WorkgroupSizeLog2].weightedDir.x;
    }
};

struct DirYAccessor
{
    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
      value = preloadData.data[ix >> config_t::WorkgroupSizeLog2].weightedDir.y;
    }
};

struct DirZAccessor
{
    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
      value = preloadData.data[ix >> config_t::WorkgroupSizeLog2].weightedDir.z;
    }
};

struct LumaAccessor
{
    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
      value = preloadData.data[ix >> config_t::WorkgroupSizeLog2].luma;
    }
};

[numthreads(config_t::WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t localInvocationIndex : SV_GroupIndex, uint32_t3 groupID: SV_GroupID)
{	
    ScratchProxy scratchAccessor;
    
    preloadData.preload();
    preloadData.workgroupExecutionAndMemoryBarrier();

    SLumaMeasurement measurement;

    DirXAccessor dirXAccessor;
    measurement.weightedDir.x= workgroup2::reduction<config_t, plus<float32_t>, device_capabilities>::template __call<DirXAccessor, ScratchProxy>(dirXAccessor, scratchAccessor);

    DirYAccessor dirYAccessor;
    measurement.weightedDir.y = workgroup2::reduction<config_t, plus<float32_t>, device_capabilities>::template __call<DirYAccessor, ScratchProxy>(dirYAccessor, scratchAccessor);

    DirZAccessor dirZAccessor;
    measurement.weightedDir.z = workgroup2::reduction<config_t, plus<float32_t>, device_capabilities>::template __call<DirZAccessor, ScratchProxy>(dirZAccessor, scratchAccessor);

    LumaAccessor lumaAccessor;
    measurement.luma = workgroup2::reduction<config_t, plus<float32_t>, device_capabilities>::template __call<LumaAccessor, ScratchProxy>(lumaAccessor, scratchAccessor);
    
    measurement.maxLuma = workgroup2::reduction<config_t, maximum<float32_t>, device_capabilities>::template __call<LumaAccessor, ScratchProxy>(lumaAccessor, scratchAccessor);
    
    if (localInvocationIndex == 0) 
      vk::RawBufferStore<SLumaMeasurement>(pc.lumaMeasurementBuf + (groupID.x * sizeof(SLumaMeasurement)), measurement);
}
