#include "IrrlichtDevice.h"

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

#include "irr/asset/IAssetManager.h"

namespace irr
{

IrrlichtDevice::IrrlichtDevice() : m_assetMgr{nullptr}
{
}

IrrlichtDevice::~IrrlichtDevice()
{
    if (m_assetMgr)
        delete m_assetMgr;
}

asset::IAssetManager& IrrlichtDevice::getAssetManager()
{
    if (!m_assetMgr)
    {
        m_assetMgr = new asset::IAssetManager(getFileSystem());
        addLoadersAndWriters();
    }
    return *m_assetMgr;
}
const asset::IAssetManager& IrrlichtDevice::getAssetManager() const
{
    return const_cast<IrrlichtDevice*>(this)->getAssetManager();
}

void IrrlichtDevice::addLoadersAndWriters()
{
#ifdef _IRR_COMPILE_WITH_STL_LOADER_
    {
        auto ldr = new asset::CSTLMeshFileLoader();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_PLY_LOADER_
    {
        auto ldr = new asset::CPLYMeshFileLoader(getSceneManager());
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_X_LOADER_
    {
        auto ldr = new asset::CXMeshFileLoader(this);
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_
    {
        auto ldr = new asset::COBJMeshFileLoader(this);
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_BAW_LOADER_
    {
        auto ldr = new asset::CBAWMeshFileLoader(this);
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_BMP_LOADER_
    {
        auto ldr = new asset::CImageLoaderBMP();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_DDS_LOADER_
    {
        auto ldr = new asset::CImageLoaderDDS();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_JPG_LOADER_
    {
        auto ldr = new asset::CImageLoaderJPG();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_PNG_LOADER_
    {
        auto ldr = new asset::CImageLoaderPng();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_TGA_LOADER_
    {
        auto ldr = new asset::CImageLoaderTGA();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_BAW_WRITER_
    {
        auto wtr = new asset::CBAWMeshWriter(getFileSystem());
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_PLY_WRITER_
    {
        auto wtr = new asset::CPLYMeshWriter();
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_STL_WRITER_
    {
        auto wtr = new asset::CSTLMeshWriter(getSceneManager());
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_TGA_WRITER_
    {
        auto wtr = new asset::CImageWriterTGA();
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_JPG_WRITER_
    {
        auto wtr = new asset::CImageWriterJPG();
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_PNG_WRITER_
    {
        auto wtr = new asset::CImageWriterPNG();
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_BMP_WRITER_
    {
        auto wtr = new asset::CImageWriterBMP();
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
}

}