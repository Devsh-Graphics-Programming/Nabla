#ifndef _NBL_CGENERAL_PURPOSE_GIMBAL_HPP_
#define _NBL_CGENERAL_PURPOSE_GIMBAL_HPP_

#include "IGimbal.hpp"

namespace nbl::core
{
    /// @brief Minimal concrete gimbal wrapper for code that only needs the generic `IGimbal` behavior.
    ///
    /// The class exists mainly as a convenient instantiable type when no additional
    /// camera-specific state or manipulation policy is required on top of `IGimbal`.
    template<typename T = hlsl::float64_t>
    class CGeneralPurposeGimbal : public IGimbal<T>
    {
    public:
        using base_t = IGimbal<T>;

        /// @brief Construct the gimbal from an initial world-space pose.
        CGeneralPurposeGimbal(typename base_t::SCreationParameters&& parameters) : base_t(std::move(parameters)) {}
        ~CGeneralPurposeGimbal() = default;
    };
}

#endif // _NBL_CGENERAL_PURPOSE_GIMBAL_HPP_
