// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_DISPATCH_FACTORIZATION_H_INCLUDED__
#define __NBL_CORE_DISPATCH_FACTORIZATION_H_INCLUDED__

#include <nabla.h>

namespace nbl
{
namespace core
{

core::vector3du32_SIMD factorizeDispatch(size_t origCnt)
{
	constexpr uint32_t limit = 64000u;

	if (origCnt < limit)
		return core::vector3du32_SIMD(origCnt, 1u, 1u);

	const uint32_t s = (origCnt + limit - 1u) / limit;

	if(s > limit)
		return core::vector3du32_SIMD(0u, 0u, 0u);

	uint32_t leastProduct = std::numeric_limits<uint32_t>::max();
	core::vector3du32_SIMD output(0u);

	for (size_t i = s; i <= limit; i++)
	{
		size_t x = std::ceil(static_cast<double>(origCnt) / static_cast<double>(i));
		if (x * i == origCnt)
			return core::vector3du32_SIMD(x, i, 1u);

		if(x * i - origCnt < 20)
			std::cout << x * i - origCnt << std::endl;

		if (x * i < leastProduct)
		{
			leastProduct = x * i;
			output = core::vector3du32_SIMD(x, i, 1u);
		}
	}

	return output;
}

}
}

#endif