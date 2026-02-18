#ifndef _NBL_BUILTIN_HLSL_SAMPLING_WARP_SPHERICAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_WARP_SPHERICAL_INCLUDED_

#include <nbl/builtin/hlsl/numbers.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/sampling/warp.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{
namespace warp
{

template <typename T = float32_t>
struct Spherical 
{
	using scalar_type = T;
	using domain_type = vector<scalar_type, 2>;
	using codomain_type = vector<scalar_type, 3>;

	template <typename DomainT NBL_FUNC_REQUIRES(is_same_v<DomainT, domain_type>)
	static WarpResult<codomain_type> warp(const DomainT uv)
	{
		codomain_type dir;
		dir.x = cos(uv.x * scalar_type(2) * numbers::pi<scalar_type>);
		dir.z = sqrt(scalar_type(1) - (dir.x * dir.x));
    if (uv.x > scalar_type(0.5))
			dir.z = -dir.z;
		const scalar_type theta = uv.y * numbers::pi<scalar_type>;
		const scalar_type cosTheta = cos(theta);
		const scalar_type sinTheta = sqrt(scalar_type(1) - (cosTheta * cosTheta));
		dir.xz *= sinTheta;
		dir.y = cosTheta;

		WarpResult<codomain_type> warpResult;
		warpResult.dst = dir;
		warpResult.density = scalar_type(1) / (scalar_type(2) * sinTheta * numbers::pi<scalar_type> * numbers::pi<scalar_type>);

		return warpResult;
	}

	template <typename DomainT NBL_FUNC_REQUIRES(is_same_v<DomainT, domain_type>)
	static float32_t2 warp2(const DomainT uv)
	{
		const scalar_type phi = scalar_type(2) * uv.x * numbers::pi<scalar_type> - numbers::pi<scalar_type>;
		const scalar_type theta = uv.y * numbers::pi<scalar_type>;
		return float32_t2(phi, theta);
	}

	template <typename CodomainT NBL_FUNC_REQUIRES(is_same_v<CodomainT, codomain_type>) 
	static domain_type inverseWarp(const CodomainT v)
	{
		const scalar_type phi = atan2(v.z, v.x);
		const scalar_type theta = acos(v.y);
		scalar_type uv_x = phi * scalar_type(0.5) * numbers::inv_pi<scalar_type>;
		if (uv_x < scalar_type(0))
			uv_x += scalar_type(1);
		scalar_type uv_y = theta * numbers::inv_pi<scalar_type>;
    return domain_type(uv_x, uv_y);
	}


	template <typename DomainT NBL_FUNC_REQUIRES(is_same_v<DomainT, domain_type>)
	static scalar_type forwardDensity(const DomainT uv)
	{
		const scalar_type theta = uv.y * numbers::pi<scalar_type>;
		return scalar_type(1) / (sin(theta) * scalar_type(2) * numbers::pi<scalar_type> * numbers::pi<scalar_type>);

	}

	template <typename CodomainT NBL_FUNC_REQUIRES(is_same_v<CodomainT, codomain_type>)
	static scalar_type backwardDensity(const CodomainT dst)
	{
		const scalar_type cosTheta = dst.y;
		const scalar_type sinTheta = sqrt(scalar_type(1) - (cosTheta * cosTheta));
		return scalar_type(1) / (sinTheta * scalar_type(2) * numbers::pi<scalar_type> * numbers::pi<scalar_type>);
	}
};

}
}
}
}

#endif