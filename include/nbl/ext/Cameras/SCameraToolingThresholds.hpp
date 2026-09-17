// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_CAMERAS_S_CAMERA_TOOLING_THRESHOLDS_HPP_
#define _NBL_EXT_CAMERAS_S_CAMERA_TOOLING_THRESHOLDS_HPP_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl::ext::cameras
{

/// @brief Comparison thresholds used by the tooling layers (goal solver, presets, follow, scripted checks) outside the runtime camera interface.
///
/// TODO (review of the tooling clusters): classify every constant as a numerical guard, a policy floor or a comparison tolerance
/// and justify its value; several of them double as divide-by-zero guards today.
struct SCameraToolingThresholds final
{
    /// @brief Default scalar tolerance used by typed state comparisons.
    static inline constexpr hlsl::float64_t ScalarTolerance = 1e-6;
    /// @brief Small epsilon used by replay and comparison helpers that need stricter zero tests.
    static inline constexpr hlsl::float64_t TinyScalarEpsilon = 1e-9;
    /// @brief Default world-space position tolerance used by pose comparisons.
    static inline constexpr hlsl::float64_t DefaultPositionTolerance = 2.0 * ScalarTolerance;
    /// @brief Default angular tolerance in degrees used by pose and state comparisons.
    static inline constexpr hlsl::float64_t DefaultAngularToleranceDeg = 1e-1;
};

} // namespace nbl::ext::cameras

#endif // _NBL_EXT_CAMERAS_S_CAMERA_TOOLING_THRESHOLDS_HPP_
