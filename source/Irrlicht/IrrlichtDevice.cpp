// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "IrrlichtDevice.h"

#include "irr/asset/IAssetManager.h"

namespace irr
{

IrrlichtDevice::IrrlichtDevice() : m_assetMgr()
{
}

IrrlichtDevice::~IrrlichtDevice()
{
}

asset::IAssetManager* IrrlichtDevice::getAssetManager()
{
    if (!m_assetMgr) // this init is messed up
        m_assetMgr = core::make_smart_refctd_ptr<asset::IAssetManager>(core::smart_refctd_ptr<io::IFileSystem>(getFileSystem()));
    return m_assetMgr.get();
}
const asset::IAssetManager* IrrlichtDevice::getAssetManager() const
{
    return const_cast<IrrlichtDevice*>(this)->getAssetManager();
}

}