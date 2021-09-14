// Copyright (C) 2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/system/IFile.h"
#include "nbl/system/ISystem.h"

using namespace nbl::system;

IFile::IFile(core::smart_refctd_ptr<ISystem>&& _system, core::bitflag<E_CREATE_FLAGS> _flags) : m_system(std::move(_system)), m_flags(_flags)
{
}

void IFile::read(future<size_t>& fut, void* buffer, size_t offset, size_t sizeToRead)
{
	m_system->readFile(fut, this, buffer, offset, sizeToRead);
}

void IFile::write(future<size_t>& fut, const void* buffer, size_t offset, size_t sizeToWrite)
{
	m_system->writeFile(fut, this, buffer, offset, sizeToWrite);
}
#include "nbl/core/definitions.h"