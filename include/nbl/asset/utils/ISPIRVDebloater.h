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
            core::smart_refctd_ptr<ICPUBuffer> spirv; // nullptr if there is some entry point not found or spirv does not need to be debloated
            bool isSuccess;

            inline operator bool() const
            {
                return isSuccess;
            }
        };

        struct EntryPoint
        {
            std::string_view name;
            hlsl::ShaderStage stage;

            inline bool operator==(const EntryPoint& rhs) const
            {
              if (stage != rhs.stage) return false;
              return name == rhs.name;
            }

            inline auto operator<=>(const EntryPoint& other) const
            {
                if (auto cmp = stage <=> other.stage; cmp != 0)
                    return cmp;
                return name <=> other.name;
            }
        };

        Result debloat(const ICPUBuffer* spirvBuffer, const core::set<EntryPoint>& entryPoints, system::logger_opt_ptr logger = nullptr) const;

        inline core::smart_refctd_ptr<const IShader> debloat(const IShader* shader, const core::set<EntryPoint>& entryPoints, system::logger_opt_ptr logger = nullptr) const
        {
            if (shader->getContentType() != IShader::E_CONTENT_TYPE::ECT_SPIRV)
            {
                logger.log("shader content must be spirv!", system::ILogger::ELL_ERROR);
                return nullptr;
            }
            const auto buffer = shader->getContent();
            const auto result = debloat(buffer, entryPoints, logger);
            if (result && result.spirv.get() == nullptr)
            {
                // when debloat does not happen return original shader
                return core::smart_refctd_ptr<const IShader>(shader);
            }

            return core::make_smart_refctd_ptr<IShader>(core::smart_refctd_ptr(result.spirv), shader->getContentType(), std::string(shader->getFilepathHint()));
        }

    private:
        core::smart_refctd_ptr<ISPIRVOptimizer> m_optimizer;
};

}

#endif
