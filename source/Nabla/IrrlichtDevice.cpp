// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "IrrlichtDevice.h"

#include "nbl/asset/IAssetManager.h"

namespace nbl
{
IrrlichtDevice::IrrlichtDevice()
    : m_assetMgr()
{
}

IrrlichtDevice::~IrrlichtDevice()
{
}

asset::IAssetManager* IrrlichtDevice::getAssetManager()
{
    if(!m_assetMgr)  // this init is messed up
        m_assetMgr = core::make_smart_refctd_ptr<asset::IAssetManager>(core::smart_refctd_ptr<io::IFileSystem>(getFileSystem()));
    return m_assetMgr.get();
}
const asset::IAssetManager* IrrlichtDevice::getAssetManager() const
{
    return const_cast<IrrlichtDevice*>(this)->getAssetManager();
}

}