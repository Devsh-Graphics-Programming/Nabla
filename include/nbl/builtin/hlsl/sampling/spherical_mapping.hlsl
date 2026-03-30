#ifndef _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_MAPPING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_MAPPING_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template <typename T = float32_t>
struct SphericalMapping 
{
  using scalar_type = T;
  using vector2_type = vector<T, 2>;
  using vector3_type = vector<T, 3>;
  using vector4_type = vector<T, 4>;

  using density_type = scalar_type;
  using weight_type = scalar_type;
  using domain_type = vector2_type;
  using codomain_type = vector3_type;

  struct cache_type
  {
    scalar_type sinTheta;
  };

  static codomain_type generate(const domain_type uv, NBL_REF_ARG(cache_type) cache)
  {
    codomain_type dir;
    dir.x = cos(uv.x * scalar_type(2) * numbers::pi<scalar_type>);
    dir.z = sqrt(scalar_type(1) - (dir.x * dir.x));
    if (uv.x > scalar_type(0.5))
      dir.z = -dir.z;
    const scalar_type theta = uv.y * numbers::pi<scalar_type>;
    scalar_type sinTheta, cosTheta;
    nbl::hlsl::math::sincos<float>(theta, sinTheta, cosTheta);
    dir.xz *= sinTheta;
    dir.y = cosTheta;

    cache.sinTheta = sinTheta;

    return dir;
  }

  static domain_type generateInverse(const codomain_type v)
  {
    const density_type phi = atan2(v.z, v.x);
    const density_type theta = acos(v.y);
    density_type uv_x = phi * density_type(0.5) * numbers::inv_pi<density_type>;
    if (uv_x < density_type(0))
      uv_x += density_type(1);
    density_type uv_y = theta * numbers::inv_pi<density_type>;
    return domain_type(uv_x, uv_y);
  }

  density_type forwardPdf(const domain_type v, const cache_type cache)
  {
    return scalar_type(1) / (scalar_type(2) * cache.sinTheta * numbers::pi<scalar_type> *numbers::pi<scalar_type>);
  }

  weight_type forwardWeight(const domain_type v, const cache_type cache)
  {
    return scalar_type(1) / (scalar_type(2) * cache.sinTheta * numbers::pi<scalar_type> *numbers::pi<scalar_type>);
  }

  density_type backwardPdf(const codomain_type v)
  {
    const density_type cosTheta = v.y;
    const density_type rcpSinTheta = hlsl::rsqrt(density_type(1) - (cosTheta * cosTheta));
    return rcpSinTheta / (density_type(2) * numbers::pi<density_type> * numbers::pi<density_type>);
  }

  weight_type backwardWeight(const codomain_type v)
  {
    const density_type cosTheta = v.y;
    const density_type rcpSinTheta = hlsl::rsqrt(density_type(1) - (cosTheta * cosTheta));
    return rcpSinTheta / (density_type(2) * numbers::pi<density_type> * numbers::pi<density_type>);
  }
};

}
}
}

#endif
