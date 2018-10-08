#ifndef __IRR_ASSET_TRAITS_H_INCLUDED__
#define __IRR_ASSET_TRAITS_H_INCLUDED__

#include "IGPUBuffer.h"
#include "ITexture.h"
#include "IMeshBuffer.h"
#include "IMesh.h"

namespace irr { namespace asset
{

template<typename AssetType>
struct asset_traits;

template<>
struct asset_traits<core::ICPUBuffer> { using GPUObjectType = video::IGPUBuffer; };
template<>
struct asset_traits<scene::ICPUMeshBuffer> { using GPUObjectType = scene::IGPUMeshBuffer; };
template<>
struct asset_traits<scene::ICPUMesh> { using GPUObjectType = scene::IGPUMesh; };
template<>
struct asset_traits<asset::ICPUTexture> { using GPUObjectType = video::ITexture; };

}}

#endif //__IRR_ASSET_TRAITS_H_INCLUDED__