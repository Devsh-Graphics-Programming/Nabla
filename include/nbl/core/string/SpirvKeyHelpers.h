#ifndef _NBL_CORE_STRING_SPIRV_KEY_HELPERS_H_INCLUDED_
#define _NBL_CORE_STRING_SPIRV_KEY_HELPERS_H_INCLUDED_

#include <cstddef>
#include <type_traits>
#include <concepts>
#include <tuple>

#include "nbl/core/string/StringLiteral.h"

namespace nbl::core::detail
{

template<nbl::core::StringLiteral>
struct SpirvKeyBuilderMissing : std::false_type {};

template<nbl::core::StringLiteral Key>
struct SpirvKeyBuilder
{
	template<typename... Args>
	static constexpr void build(const Args&...)
	{
		static_assert(SpirvKeyBuilderMissing<Key>::value, "Unknown SPIR-V key");
	}
};

template<class Device>
concept spirv_device_has_limits = requires(const Device* device)
{
	device->getPhysicalDevice()->getLimits();
};

template<class Device>
concept spirv_device_has_features = requires(const Device* device)
{
	device->getEnabledFeatures();
};

template<class Device>
constexpr decltype(auto) spirv_device_get_limits(const Device* device)
{
	static_assert(spirv_device_has_limits<Device>, "Device does not provide getLimits");
	return device->getPhysicalDevice()->getLimits();
}

template<class Device>
constexpr decltype(auto) spirv_device_get_features(const Device* device)
{
	static_assert(spirv_device_has_features<Device>, "Device does not provide getEnabledFeatures");
	return device->getEnabledFeatures();
}

}

#endif
