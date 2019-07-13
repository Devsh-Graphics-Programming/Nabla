# Copyright (c) 2019 DevSH Graphics Programming Sp. z O.O.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


if (IRR_BUILD_CEGUI)
    set(_OLD_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH ${CEGUI_DEPS_INSTALL_PATH})
    #message(FATAL_ERROR "${CEGUI_DEPS_INSTALL_PATH}") # CEGUI_DEPS_INSTALL_PATH is empty forward it around somehow
    add_subdirectory(CEGUI)
    set(IRR_EXT_CEGUI_INCLUDE_DIRS
        ${IRR_EXT_CEGUI_INCLUDE_DIRS}
        PARENT_SCOPE
    )
    set(IRR_EXT_CEGUI_LIB
        ${IRR_EXT_CEGUI_LIB}
        PARENT_SCOPE
    )
    set(CMAKE_PREFIX_PATH ${_OLD_CMAKE_PREFIX_PATH})
endif()

if (IRR_BUILD_BULLET)
    add_subdirectory(Bullet)
    set(IRR_EXT_BULLET_INCLUDE_DIRS
        ${IRR_EXT_BULLET_INCLUDE_DIRS}
        PARENT_SCOPE
    )
    set(IRR_EXT_BULLET_LIB
        ${IRR_EXT_BULLET_LIB}
        PARENT_SCOPE
    )
endif()
