#ifndef _NBL_I_CPU_RENDERPASS_H_INCLUDED_
#define _NBL_I_CPU_RENDERPASS_H_INCLUDED_

#include "nbl/asset/IAsset.h"
#include "nbl/asset/IRenderpass.h"

namespace nbl::asset
{

class ICPURenderpass : public IRenderpass, public IAsset
{
    public:
        static inline core::smart_refctd_ptr<ICPURenderpass> create(const SCreationParams& _params)
        {
            const SCreationParamValidationResult validation = validateCreationParams(_params);
            if (!validation)
                return nullptr;

            return core::smart_refctd_ptr<ICPURenderpass>(new ICPURenderpass(_params, validation), core::dont_grab);
        }

        inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            return core::smart_refctd_ptr<ICPURenderpass>(new ICPURenderpass(m_params,SCreationParamValidationResult{
                .depthStencilAttachmentCount = m_depthStencilAttachments ? static_cast<uint32_t>(m_depthStencilAttachments->size()):0u,
                .colorAttachmentCount = m_colorAttachments ? static_cast<uint32_t>(m_colorAttachments->size()):0u,
                .subpassCount = m_subpasses ? static_cast<uint32_t>(m_subpasses->size()):0u,
                .totalInputAttachmentCount = m_inputAttachments ? static_cast<uint32_t>(m_inputAttachments->size()):0u,
                .totalPreserveAttachmentCount = m_preserveAttachments ? static_cast<uint32_t>(m_preserveAttachments->size()):0u,
                .dependencyCount = m_subpassDependencies ? static_cast<uint32_t>(m_subpassDependencies->size()):0u,
                .viewMaskMSB = m_viewMaskMSB,
            }),core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_RENDERPASS;
        E_TYPE getAssetType() const override
        {
            return ET_RENDERPASS;
        }

        inline core::unordered_set<const IAsset*> computeDependants() const override
        {
            return {};
        }

    protected:
        inline ICPURenderpass(const SCreationParams& _params, const SCreationParamValidationResult& _validation) : IRenderpass(_params, _validation) {}
        inline ~ICPURenderpass() = default;

};

}
#endif