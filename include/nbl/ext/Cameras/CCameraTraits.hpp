// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_TRAITS_HPP_
#define _C_CAMERA_TRAITS_HPP_

#include <limits>

namespace nbl::core
{

/// @brief Geometric constants used by target-relative camera families.
///
/// `MinDistance` prevents zero-distance target-relative states.
/// `DefaultMaxDistance` is unbounded. Individual cameras and tools may apply
/// their own finite limits on top of it.
struct SCameraTargetRelativeTraits final
{
    /// @brief Smallest valid target-relative distance shared by spherical and path-style rigs.
    static inline constexpr float MinDistance = 0.1f;
    /// @brief Default upper bound for target-relative distance when no camera-specific cap is requested.
    static inline constexpr float DefaultMaxDistance = std::numeric_limits<float>::infinity();
};

/// @brief Comparison thresholds used by helper layers outside the runtime camera interface.
struct SCameraToolingThresholds final
{
    /// @brief Default scalar tolerance used by typed state comparisons.
    static inline constexpr double ScalarTolerance = 1e-6;
    /// @brief Small epsilon used by replay and comparison helpers that need stricter zero tests.
    static inline constexpr double TinyScalarEpsilon = 1e-9;
    /// @brief Default world-space position tolerance used by pose comparisons.
    static inline constexpr double DefaultPositionTolerance = 2.0 * ScalarTolerance;
    /// @brief Default angular tolerance in degrees used by pose and state comparisons.
    static inline constexpr double DefaultAngularToleranceDeg = 0.1;
};

} // namespace nbl::core

#endif // _C_CAMERA_TRAITS_HPP_
