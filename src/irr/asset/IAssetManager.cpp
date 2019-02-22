#include "irr/asset/IAssetManager.h"
#include "irr/asset/CGeometryCreator.h"
#include "irr/asset/CMeshManipulator.h"

using namespace irr;
using namespace asset;

std::function<void(IAsset*)> irr::asset::makeAssetGreetFunc(const IAssetManager* const _mgr)
{
    return [_mgr](IAsset* _asset) { _asset->grab(); _mgr->setAssetCached(_asset, true); };
}
std::function<void(IAsset*)> irr::asset::makeAssetDisposeFunc(const IAssetManager* const _mgr)
{
    return [_mgr](IAsset* _asset) { _mgr->setAssetCached(_asset, false); _asset->drop(); };
}

void IAssetManager::initializeMeshTools()
{
    m_geometryCreator = new CGeometryCreator();
    m_meshManipulator = new CMeshManipulator();
}

void IAssetManager::dropMeshTools()
{
    m_geometryCreator->drop();
    m_meshManipulator->drop();
}

const IGeometryCreator* IAssetManager::getGeometryCreator() const
{
    return m_geometryCreator;
}

const IMeshManipulator* IAssetManager::getMeshManipulator() const
{
    return m_meshManipulator;
}
