#ifndef _NBL_ASSET_I_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_GRAPHICS_PIPELINE_H_INCLUDED_


#include "nbl/asset/IShader.h"
#include "nbl/asset/RasterizationStates.h"
#include "nbl/asset/IPipeline.h"
#include "nbl/asset/IRenderpass.h"

#include <span>


namespace nbl::asset
{

struct SVertexInputAttribParams
{
    inline auto operator<=>(const SVertexInputAttribParams& rhs) const = default;

    uint32_t binding : 4 = 0;
    uint32_t format : 8  = EF_UNKNOWN; // asset::E_FORMAT
    uint32_t relativeOffset : 13 = 0; // assuming max=2048
};
static_assert(sizeof(SVertexInputAttribParams)==4u, "Unexpected size!");

struct SVertexInputBindingParams
{
    enum E_VERTEX_INPUT_RATE : uint32_t
    {
        EVIR_PER_VERTEX = 0,
        EVIR_PER_INSTANCE = 1
    };

    inline auto operator<=>(const SVertexInputBindingParams& rhs) const = default;

    uint32_t stride : 31 = 0u;
    E_VERTEX_INPUT_RATE inputRate : 1 = EVIR_PER_VERTEX;
};
static_assert(sizeof(SVertexInputBindingParams)==4u, "Unexpected size!");

struct SVertexInputParams
{
    constexpr static inline size_t MAX_VERTEX_ATTRIB_COUNT = 16u;
    constexpr static inline size_t MAX_ATTR_BUF_BINDING_COUNT = 16u;

    inline auto operator<=>(const SVertexInputParams& rhs) const = default;


    uint16_t enabledAttribFlags = 0u;
    uint16_t enabledBindingFlags = 0u;
    //! index in array denotes location (attribute ID)
	SVertexInputAttribParams attributes[MAX_VERTEX_ATTRIB_COUNT] = {};
    //! index in array denotes binding number
	SVertexInputBindingParams bindings[MAX_ATTR_BUF_BINDING_COUNT] = {};
};
static_assert(sizeof(SVertexInputParams)==(sizeof(uint16_t)*2u+SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT*sizeof(SVertexInputAttribParams)+SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT*sizeof(SVertexInputBindingParams)),"Unexpected size!");

struct SPrimitiveAssemblyParams
{
    inline auto operator<=>(const SPrimitiveAssemblyParams& other) const = default;

    E_PRIMITIVE_TOPOLOGY primitiveType : 5 = EPT_TRIANGLE_LIST;
    uint16_t primitiveRestartEnable : 1 = false;
    uint16_t tessPatchVertCount : 10 = 3u;
};
static_assert(sizeof(SPrimitiveAssemblyParams)==2u, "Unexpected size!");


class IGraphicsPipelineBase : public virtual core::IReferenceCounted
{
    public:
        constexpr static inline uint8_t GRAPHICS_SHADER_STAGE_COUNT = 5u;

        struct SCachedCreationParams final
        {
            SVertexInputParams vertexInput = {};
            SPrimitiveAssemblyParams primitiveAssembly = {};
            SRasterizationParams rasterization = {};
            SBlendParams blend = {};
            uint32_t subpassIx = 0u;
        };
};

template<typename PipelineLayoutType, typename RenderpassType>
class IGraphicsPipeline : public IPipeline<PipelineLayoutType>, public IGraphicsPipelineBase
{
    protected:
        using renderpass_t = RenderpassType;

    public:
        inline const SCachedCreationParams& getCachedCreationParams() const {return m_params;}
        inline const renderpass_t* getRenderpass() const {return m_renderpass.get();}

        static inline int8_t stageToIndex(const hlsl::ShaderStage stage)
        {
            const auto stageIx = hlsl::findLSB(stage);
            if (stageIx < 0 || stageIx >= GRAPHICS_SHADER_STAGE_COUNT || hlsl::bitCount(stage)!=1)
              return -1;
            return stageIx;
        }

        static inline hlsl::ShaderStage indexToStage(const int8_t index)
        {
            if (index < 0 || index > GRAPHICS_SHADER_STAGE_COUNT)
                return hlsl::ShaderStage::ESS_UNKNOWN;
            return static_cast<hlsl::ShaderStage>(hlsl::ShaderStage::ESS_VERTEX + index);
        }

        static inline bool isValidStagePresence(const core::bitflag<hlsl::ShaderStage>& stagePresence, E_PRIMITIVE_TOPOLOGY primitiveType)
        {
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-stage-02096
            if (!stagePresence.hasFlags(hlsl::ShaderStage::ESS_VERTEX))
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-pStages-00729
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-pStages-00730
            if (stagePresence.hasFlags(hlsl::ShaderStage::ESS_TESSELLATION_CONTROL)!=stagePresence.hasFlags(hlsl::ShaderStage::ESS_TESSELLATION_EVALUATION))
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-pStages-08888
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-topology-08889
            if (stagePresence.hasFlags(hlsl::ShaderStage::ESS_TESSELLATION_EVALUATION)!=(primitiveType==asset::EPT_PATCH_LIST))
                return false;
            return true;
        }

    protected:
        explicit IGraphicsPipeline(const PipelineLayoutType* layout, const SCachedCreationParams& cachedParams, const renderpass_t* renderpass) :
            IPipeline<PipelineLayoutType>(core::smart_refctd_ptr<const PipelineLayoutType>(layout)), m_renderpass(core::smart_refctd_ptr<renderpass_t>(renderpass))
        {}

        SCachedCreationParams m_params = {};
        core::smart_refctd_ptr<renderpass_t> m_renderpass = nullptr;
};

}

#endif