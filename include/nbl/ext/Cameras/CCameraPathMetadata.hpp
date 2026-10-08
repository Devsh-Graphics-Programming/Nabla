// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PATH_METADATA_HPP_
#define _C_CAMERA_PATH_METADATA_HPP_

#include <string_view>

namespace nbl::ext::cameras
{

/// @brief Stable descriptive strings used by the reusable `Path Rig` camera kind.
///
/// This metadata lives in a lightweight header so code that only needs labels
/// or identifiers does not have to include the full path-model implementation.
struct SCameraPathRigMetadata final
{
    /// @brief User-facing camera kind label.
    static inline constexpr std::string_view KindLabel = "Path Rig";
    /// @brief Short user-facing description of the camera kind.
    static inline constexpr std::string_view KindDescription = "Path-model camera with typed s/u/v/roll state";
    /// @brief Default runtime identifier used by the concrete camera instance.
    static inline constexpr std::string_view Identifier = "Target-relative Path Rig";
    /// @brief Default description of the built-in path model shipped by the shared API.
    static inline constexpr std::string_view DefaultModelDescription = "Adjust a target-relative path rig with s/u/v/roll state";
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_PATH_METADATA_HPP_
