# Nabla Style Guide

This document describes the naming, layout and formatting conventions used in Nabla.

## Naming

| Thing | Form | Example |
| --- | --- | --- |
| C preprocessor macro | `CAPITAL_SNAKE_CASE`, `NBL_` prefix  | `NBL_CONSTEXPR_STATIC_INLINE`, `NBL_REF_ARG` |
| Internal / private macro | `_NBL_` prefix | `_NBL_STATIC_INLINE_CONSTEXPR`, `_NBL_DEBUG` |
| Include guard | `_NBL_<PATH>_<FILE>_H_INCLUDED_` | `_NBL_VIDEO_I_GPU_BUFFER_H_INCLUDED_` |
| constexpr / compile time constant | `CamelCase`, first letter capital  | `MaxFramesInFlight`, `WorkgroupSize`, `IsAnisotropic` |
| Interface class | `I` prefix + CamelCase | `IGPUBuffer`, `ILogicalDevice` |
| Concrete class | `C` prefix + CamelCase | `CVulkanLogicalDevice`, `CAssetConverter` |
| Copy by value struct | `S` prefix + CamelCase | `SCreationParams`, `SBufferBinding` |
| Enum type | `E` prefix + CamelCase, `enum class`, explicit underlying type | `enum class ECreateFlags : uint8_t` |
| Enum value | `CamelCase`, `Bit` suffix for flag bits | `TransientBit`, `None` |
| Function / method | `lowerCamelCase` | `getDeviceAddress`, `createBuffer` |
| Non static data member | `m_` + lowerCamelCase | `m_creationParams`, `m_logger` |
| Parameter, local variable | `lowerCamelCase` | `handleCount`, `memoryPropertyFlags` |
| Template type parameter | CamelCase or single capital | `T`, `AssetType`, `BufferType` |
| Namespace | lowercase and short, `snake_case` only when a single word cannot be found | `nbl::core`, `nbl::asset`, `nbl::video`, `nbl::hlsl` |
| C++ header / source file | named after the main class it declares, `.hpp` for new headers | `IGPUBuffer.hpp`, `CVulkanLogicalDevice.cpp` |
| HLSL builtin file | `snake_case.hlsl` | `type_traits.hlsl`, `cos_weighted_spheres.hlsl` |
| STL mirroring HLSL trait / alias | `snake_case`, `_t` / `_v` suffix | `is_scalar_v`, `alignment_of`, `vector_traits` |

Legacy forms you will still see, inherited from Irrlicht. Do not add new ones:

- Enum types in `CAPITAL_SNAKE_CASE` (`enum class CREATE_FLAGS`, `enum E_FORMAT`) and enum values in `CAPITAL_SNAKE_CASE` (`TRANSIENT_BIT`, `EF_R8G8B8A8_UNORM`).
- `.h` headers. New headers are `.hpp`. Mixing `.h` and `.hpp` is allowed for now.

## Formatting

- Allman braces. Namespace bodies not indented. Access specifiers indented one level, members two.
- Indentation is tabs in some files and four spaces in others. Match the file, never mix.
- Star and ampersand attach to the type, `const` comes before the type: `IGPUBuffer* buffer`, `const SRange& range`. Not `IGPUBuffer *buffer` or `SRange const& range`.
- `public`, then `protected`, then `private` is the general recommendation, not a hard rule. Nested types and forward references sometimes force another order.

## Comments

- Doxygen comments use `//!`. Multi line blocks use `/** */`.
- Say why, or cite the spec or VUID. Do not narrate what the code does.
