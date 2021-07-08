// Copyright (C) 2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/system/IFile.h"
#include "nbl/system/ISystem.h"

using namespace nbl::system;

IFile::IFile(core::smart_refctd_ptr<ISystem>&& _system, std::underlying_type_t<E_CREATE_FLAGS> _flags) : m_system(std::move(_system)), m_flags(_flags)
{
}

#include "nbl/core/definitions.h"