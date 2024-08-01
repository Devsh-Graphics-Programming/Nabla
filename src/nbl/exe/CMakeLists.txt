# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

add_subdirectory(xxHash256)

NBL_GET_ALL_TARGETS(NBL_ALL_TARGETS)
foreach(NBL_TARGET IN LISTS NBL_ALL_TARGETS)
	set_target_properties("${NBL_TARGET}" PROPERTIES FOLDER exe)
endforeach()