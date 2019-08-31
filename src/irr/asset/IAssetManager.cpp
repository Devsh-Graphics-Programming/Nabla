#include "irr/asset/IAssetManager.h"
#include "irr/asset/CGeometryCreator.h"
#include "irr/asset/CMeshManipulator.h"


#ifdef _IRR_COMPILE_WITH_X_LOADER_
#include "irr/asset/CXMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_
#include "irr/asset/COBJMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_STL_LOADER_
#include "irr/asset/CSTLMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_PLY_LOADER_
#include "irr/asset/CPLYMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_BAW_LOADER_
#include "irr/asset/CBAWMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_BMP_LOADER_
#include "irr/asset/CImageLoaderBMP.h"
#endif

#ifdef _IRR_COMPILE_WITH_DDS_LOADER_
#include "irr/asset/CImageLoaderDDS.h"
#endif

#ifdef _IRR_COMPILE_WITH_JPG_LOADER_
#include "irr/asset/CImageLoaderJPG.h"
#endif

#ifdef _IRR_COMPILE_WITH_PNG_LOADER_
#include "irr/asset/CImageLoaderPNG.h"
#endif

#ifdef _IRR_COMPILE_WITH_TGA_LOADER_
#include "irr/asset/CImageLoaderTGA.h"
#endif

#ifdef _IRR_COMPILE_WITH_STL_WRITER_
#include "irr/asset/CSTLMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_PLY_WRITER_
#include "irr/asset/CPLYMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_BAW_WRITER_
#include"irr/asset/CBAWMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_TGA_WRITER_
#include "irr/asset/CImageWriterTGA.h"
#endif

#ifdef _IRR_COMPILE_WITH_JPG_WRITER_
#include "irr/asset/CImageWriterJPG.h"
#endif

#ifdef _IRR_COMPILE_WITH_PNG_WRITER_
#include "irr/asset/CImageWriterPNG.h"
#endif

#ifdef _IRR_COMPILE_WITH_BMP_WRITER_
#include "irr/asset/CImageWriterBMP.h"
#endif

using namespace irr;
using namespace asset;

std::function<void(SAssetBundle&)> irr::asset::makeAssetGreetFunc(const IAssetManager* const _mgr)
{
    return [_mgr](SAssetBundle& _asset) { _mgr->setAssetCached(_asset, true); };
}
std::function<void(SAssetBundle&)> irr::asset::makeAssetDisposeFunc(const IAssetManager* const _mgr)
{
    return [_mgr](SAssetBundle& _asset) { _mgr->setAssetCached(_asset, false); };
}

void IAssetManager::initializeMeshTools()
{
    m_geometryCreator = core::make_smart_refctd_ptr<CGeometryCreator>();
    m_meshManipulator = core::make_smart_refctd_ptr<CMeshManipulator>();
}

const IGeometryCreator* IAssetManager::getGeometryCreator() const
{
	return m_geometryCreator.get();
}

const IMeshManipulator* IAssetManager::getMeshManipulator() const
{
    return m_meshManipulator.get();
}


void IAssetManager::addLoadersAndWriters()
{
#ifdef _IRR_COMPILE_WITH_STL_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CSTLMeshFileLoader>());
#endif
#ifdef _IRR_COMPILE_WITH_PLY_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CPLYMeshFileLoader>());
#endif
#ifdef _IRR_COMPILE_WITH_X_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CXMeshFileLoader>(this));
#endif
#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::COBJMeshFileLoader>(this));
#endif
#ifdef _IRR_COMPILE_WITH_BAW_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CBAWMeshFileLoader>(this));
#endif
#ifdef _IRR_COMPILE_WITH_BMP_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderBMP>());
#endif
#ifdef _IRR_COMPILE_WITH_DDS_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderDDS>());
#endif
#ifdef _IRR_COMPILE_WITH_JPG_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderJPG>());
#endif
#ifdef _IRR_COMPILE_WITH_PNG_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderPng>());
#endif
#ifdef _IRR_COMPILE_WITH_TGA_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderTGA>());
#endif
#ifdef _IRR_COMPILE_WITH_BAW_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CBAWMeshWriter>(getFileSystem()));
#endif
#ifdef _IRR_COMPILE_WITH_PLY_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CPLYMeshWriter>());
#endif
#ifdef _IRR_COMPILE_WITH_STL_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CSTLMeshWriter>());
#endif
#ifdef _IRR_COMPILE_WITH_TGA_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterTGA>());
#endif
#ifdef _IRR_COMPILE_WITH_JPG_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterJPG>());
#endif
#ifdef _IRR_COMPILE_WITH_PNG_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterPNG>());
#endif
#ifdef _IRR_COMPILE_WITH_BMP_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterBMP>());
#endif
}