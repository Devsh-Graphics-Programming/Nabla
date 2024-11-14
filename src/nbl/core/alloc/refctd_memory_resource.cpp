// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/core/alloc/refctd_memory_resource.h"

using namespace nbl;
using namespace core;

static smart_refctd_ptr<refctd_memory_resource> default_memory_resource = nullptr;

smart_refctd_ptr<refctd_memory_resource> core::getNullMemoryResource()
{
    static smart_refctd_ptr<refctd_memory_resource> null_memory_resource = nullptr;
    if (!null_memory_resource)
        null_memory_resource = make_smart_refctd_ptr<refctd_memory_resource>(std::pmr::null_memory_resource());
    return null_memory_resource;
}

smart_refctd_ptr<refctd_memory_resource> core::getDefaultMemoryResource()
{
    if (!default_memory_resource)
        default_memory_resource = make_smart_refctd_ptr<refctd_memory_resource>(std::pmr::get_default_resource());
    return default_memory_resource;
}

void core::setDefaultMemoryResource(refctd_memory_resource* memoryResource)
{
    default_memory_resource = smart_refctd_ptr<refctd_memory_resource>(memoryResource, dont_grab);
}
