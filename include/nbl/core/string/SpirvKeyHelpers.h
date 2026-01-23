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

template<nbl::core::StringLiteral Key>
struct SpirvFileKeyBuilder
{
	template<typename... Args>
	static constexpr auto build(const Args&... args)
	{
		return SpirvKeyBuilder<Key>::build(args...);
	}

	template<class Device, typename... Args>
	static constexpr auto build_from_device(const Device* device, const Args&... args)
	{
		return SpirvKeyBuilder<Key>::build_from_device(device, args...);
	}
};

template<nbl::core::StringLiteral Key, nbl::core::StringLiteral Entry>
struct SpirvEntrypointBuilder
{
	template<typename... Args>
	static constexpr void build(const Args&...)
	{
		static_assert(SpirvKeyBuilderMissing<Key>::value, "Unknown SPIR-V key");
	}

	template<class Device, typename... Args>
	static constexpr void build_from_device(const Device*, const Args&...)
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
