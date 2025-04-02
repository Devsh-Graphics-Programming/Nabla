#ifndef _NBL_ASSET_I_SPIRV_DEBLOATER_H_INCLUDED_
#define _NBL_ASSET_I_SPIRV_DEBLOATER_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUBuffer.h"

#include "nbl/system/ILogger.h"

namespace nbl::asset
{

class ISPIRVDebloater final : public core::IReferenceCounted
{
    public:
        ISPIRVDebloater();

        struct Result
        {
            bool isAllEntryPointsFound;
            core::smart_refctd_ptr<ICPUBuffer> spirv; // nullptr if there is some entry point not found or spirv does not need to be debloated
        };

        struct EntryPoint
        {
          std::string_view name;
          hlsl::ShaderStage shaderStage;

          bool operator==(const EntryPoint& rhs) const = default;
        };

        Result debloat(const ICPUBuffer* spirvBuffer, std::span<const EntryPoint> entryPoints, system::logger_opt_ptr logger) const;

    private:
        core::smart_refctd_ptr<ISPIRVOptimizer> m_optimizer;
};

}

#endif
