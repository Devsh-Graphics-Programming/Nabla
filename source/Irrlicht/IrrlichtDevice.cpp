#include "IrrlichtDevice.h"

#ifdef _IRR_COMPILE_WITH_X_LOADER_
#include "CXMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_
#include "COBJMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_STL_LOADER_
#include "CSTLMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_PLY_LOADER_
#include "CPLYMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_BAW_LOADER_
#include "CBAWMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_BMP_LOADER_
#include "CImageLoaderBMP.h"
#endif

#ifdef _IRR_COMPILE_WITH_DDS_LOADER_
#include "CImageLoaderDDS.h"
#endif

#ifdef _IRR_COMPILE_WITH_JPG_LOADER_
#include "CImageLoaderJPG.h"
#endif

#ifdef _IRR_COMPILE_WITH_PNG_LOADER_
#include "CImageLoaderPNG.h"
#endif

#ifdef _IRR_COMPILE_WITH_RGB_LOADER_
#include "CImageLoaderRGB.h"
#endif

#ifdef _IRR_COMPILE_WITH_TGA_LOADER_
#include "CImageLoaderTGA.h"
#endif

#ifdef _IRR_COMPILE_WITH_STL_WRITER_
#include "CSTLMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_PLY_WRITER_
#include "CPLYMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_BAW_WRITER_
#include"CBAWMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_TGA_WRITER_
#include "CImageWriterTGA.h"
#endif

#ifdef _IRR_COMPILE_WITH_JPG_WRITER_
#include "CImageWriterJPG.h"
#endif

#ifdef _IRR_COMPILE_WITH_PNG_WRITER_
#include "CImageWriterPNG.h"
#endif

#ifdef _IRR_COMPILE_WITH_BMP_WRITER_
#include "CImageWriterBMP.h"
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
        auto ldr = new scene::CSTLMeshFileLoader();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_PLY_LOADER_
    {
        auto ldr = new scene::CPLYMeshFileLoader(getSceneManager());
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_X_LOADER_
    {
        auto ldr = new scene::CXMeshFileLoader(this);
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_
    {
        auto ldr = new scene::COBJMeshFileLoader(this);
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_BAW_LOADER_
    {
        auto ldr = new scene::CBAWMeshFileLoader(this);
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_BMP_LOADER_
    {
        auto ldr = new video::CImageLoaderBMP();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_DDS_LOADER_
    {
        auto ldr = new video::CImageLoaderDDS();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_JPG_LOADER_
    {
        auto ldr = new video::CImageLoaderJPG();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_PNG_LOADER_
    {
        auto ldr = new video::CImageLoaderPng();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_RGB_LOADER_
    {
        auto ldr = new video::CImageLoaderRGB();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_TGA_LOADER_
    {
        auto ldr = new video::CImageLoaderTGA();
        m_assetMgr->addAssetLoader(ldr);
        ldr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_BAW_WRITER_
    {
        auto wtr = new scene::CBAWMeshWriter(getFileSystem());
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_PLY_WRITER_
    {
        auto wtr = new scene::CPLYMeshWriter();
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_STL_WRITER_
    {
        auto wtr = new scene::CSTLMeshWriter(getSceneManager());
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_TGA_WRITER_
    {
        auto wtr = new video::CImageWriterTGA();
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_JPG_WRITER_
    {
        auto wtr = new video::CImageWriterJPG();
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_PNG_WRITER_
    {
        auto wtr = new video::CImageWriterPNG();
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
#ifdef _IRR_COMPILE_WITH_BMP_WRITER_
    {
        auto wtr = new video::CImageWriterBMP();
        m_assetMgr->addAssetWriter(wtr);
        wtr->drop();
    }
#endif
}

}