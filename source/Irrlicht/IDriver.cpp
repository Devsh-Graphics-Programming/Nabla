#include "IDriver.h"

#include "IrrlichtDevice.h"
#include "IAssetManager.h"
#include "IGPUObjectFromAssetConverter.h"

namespace irr { namespace video
{

template<typename AssetType>
core::vector<typename asset::asset_traits<AssetType>::GPUObjectType*> IDriver::getGPUObjectsFromAssets(AssetType** const _begin, AssetType** const _end)
{
    return m_cpuToGpuConverter->getGPUObjectsFromAssets(_begin, _end);
}
template core::vector<typename asset::asset_traits<core::ICPUBuffer>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<core::ICPUBuffer>(core::ICPUBuffer** const, core::ICPUBuffer** const);
template core::vector<typename asset::asset_traits<scene::ICPUMeshBuffer>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<scene::ICPUMeshBuffer>(scene::ICPUMeshBuffer** const, scene::ICPUMeshBuffer** const);
template core::vector<typename asset::asset_traits<scene::ICPUMesh>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<scene::ICPUMesh>(scene::ICPUMesh** const, scene::ICPUMesh** const);
template core::vector<typename asset::asset_traits<asset::ICPUTexture>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<asset::ICPUTexture>(asset::ICPUTexture** const, asset::ICPUTexture** const);

IDriver::IDriver(IrrlichtDevice* _dev) : m_device{_dev}, m_cpuToGpuConverter{nullptr}
{
    m_cpuToGpuConverter = m_defaultCpuToGpuConverter = new IGPUObjectFromAssetConverter(&m_device->getAssetManager(), this);
}

IDriver::~IDriver()
{
    delete m_defaultCpuToGpuConverter;
}

}}

