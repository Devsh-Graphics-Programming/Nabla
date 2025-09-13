if (NOT DEFINED SPIRV_DIS_EXE)
    message(FATAL_ERROR "SPIRV_DIS_EXE is not defined.")
endif()

if (NOT DEFINED NSC_RELEASE_BUILD_INFO)
    message(FATAL_ERROR "NSC_RELEASE_BUILD_INFO is not defined.")
endif()

if (NOT DEFINED NSC_RELWITHDEBINFO_BUILD_INFO)
    message(FATAL_ERROR "NSC_RELWITHDEBINFO_BUILD_INFO is not defined.")
endif()

if (NOT DEFINED NSC_DEBUG_BUILD_INFO)
    message(FATAL_ERROR "NSC_DEBUG_BUILD_INFO is not defined.")
endif()

if (NOT DEFINED OUTPUT_CONFIG_FILE)
    message(FATAL_ERROR "OUTPUT_CONFIG_FILE is not defined.")
endif()

function(GET_HASH MODULE BUILD_INFO_JSON OUT_VAR)
    string(JSON MODULE_JSON ERROR_VARIABLE JSON_ERROR GET "${BUILD_INFO_JSON}" "modules")
    if (JSON_ERROR)
        message(FATAL_ERROR "JSON_ERROR: ${JSON_ERROR}")
    endif()

    string(JSON MODULE_HASH ERROR_VARIABLE JSON_ERROR GET "${MODULE_JSON}" "${MODULE}")
    if (JSON_ERROR)
        message(FATAL_ERROR "JSON_ERROR: ${JSON_ERROR}")
    endif()

    string(JSON HASH ERROR_VARIABLE JSON_ERROR GET "${MODULE_HASH}" "commitHash")
    if (JSON_ERROR)
        message(FATAL_ERROR "JSON_ERROR: ${JSON_ERROR}")
    endif()

    set("${OUT_VAR}" "${HASH}" PARENT_SCOPE)
endfunction()

set(NABLA_REPO_URL "https://github.com/Devsh-Graphics-Programming/Nabla")
set(DXC_REPO_URL "https://github.com/Devsh-Graphics-Programming/DirectXShaderCompiler")

function(NBL_CONFIGURE_COMPILER _CONFIG_ CONFIG_CONTENT)
set(BUILD_INFO_FILE ${NSC_${_CONFIG_}_BUILD_INFO})

message(STATUS "Configuring \"${BUILD_INFO_FILE}\"..")

if (NOT EXISTS "${BUILD_INFO_FILE}")
	message(STATUS "${_CONFIG_} compiler variant will not get created because build info file does not exist!")
	return()
endif()

file(READ "${BUILD_INFO_FILE}" BUILD_INFO_JSON)

# module hashes
GET_HASH("nabla" "${BUILD_INFO_JSON}" NABLA_COMMIT_HASH)
GET_HASH("dxc" "${BUILD_INFO_JSON}" DXC_COMMIT_HASH)

# exe
string(JSON EXE_PATH ERROR_VARIABLE JSON_ERROR GET "${BUILD_INFO_JSON}" "exe.path")
if (JSON_ERROR)
	message(FATAL_ERROR "JSON_ERROR: ${JSON_ERROR}")
endif()

string(JSON EXE_TIMESTAMP_DATE ERROR_VARIABLE JSON_ERROR GET "${BUILD_INFO_JSON}" "exe.timestamp.date")
if (JSON_ERROR)
	message(FATAL_ERROR "JSON_ERROR: ${JSON_ERROR}")
endif()

string(JSON EXE_TIMESTAMP_TIME ERROR_VARIABLE JSON_ERROR GET "${BUILD_INFO_JSON}" "exe.timestamp.time")
if (JSON_ERROR)
	message(FATAL_ERROR "JSON_ERROR: ${JSON_ERROR}")
endif()

string(TOLOWER "${_CONFIG_}" _L_CONFIG_)

set(CE_COMPILER_CONTENT 
[=[

compiler.nsc_@_L_CONFIG_@_upstream.exe=@EXE_PATH@
compiler.nsc_@_L_CONFIG_@_upstream.name=NSC (@_L_CONFIG_@)
compiler.nsc_@_L_CONFIG_@_upstream.notification=The NSC (@_L_CONFIG_@) has been compiled from the following <a href="@NABLA_REPO_URL@/commit/@NABLA_COMMIT_HASH@" target="_blank" rel="noopener noreferrer">Nabla commit<sup><small class="fas fa-external-link-alt opens-new-window" title="Opens the Nabla commit in a new window"></small></sup></a> and <a href="@DXC_REPO_URL@/commit/@DXC_COMMIT_HASH@" target="_blank" rel="noopener noreferrer">DXC commit<sup><small class="fas fa-external-link-alt opens-new-window" title="Opens the DXC commit in a new window"></small></sup></a>. @BUILD_INFO_HTML@
compiler.nsc_@_L_CONFIG_@_upstream.supportsExecute=false
compiler.nsc_@_L_CONFIG_@_upstream.options=
compiler.nsc_@_L_CONFIG_@_upstream.disassemblerPath=@SPIRV_DIS_EXE@
compiler.nsc_@_L_CONFIG_@_upstream.demangler=

]=]
)

string(REPLACE "\n" "<br>" BUILD_INFO_HTML "${BUILD_INFO_JSON}")
string(PREPEND BUILD_INFO_HTML "<br><br>Build info:<br>")

string(CONFIGURE "${CE_COMPILER_CONTENT}" CE_COMPILER_CONTENT @ONLY)
string(APPEND CONFIG_CONTENT "${CE_COMPILER_CONTENT}")
set(CONFIG_CONTENT 
	"${CONFIG_CONTENT}"
PARENT_SCOPE)

list(APPEND NSC_COMPILERS "nsc_${_L_CONFIG_}_upstream")
set(NSC_COMPILERS 
	"${NSC_COMPILERS}"
PARENT_SCOPE)

message(STATUS "OK! Target CT \"${EXE_PATH}\" configured.")
endfunction()

NBL_CONFIGURE_COMPILER(RELEASE "${CONFIG_CONTENT}" "${NSC_COMPILERS}")
NBL_CONFIGURE_COMPILER(RELWITHDEBINFO "${CONFIG_CONTENT}" "${NSC_COMPILERS}")
NBL_CONFIGURE_COMPILER(DEBUG "${CONFIG_CONTENT}" "${NSC_COMPILERS}")
string(REPLACE ";" ":" NSC_COMPILERS "${NSC_COMPILERS}")

set(FINAL_CONFIG_CONTENT
[=[
compilers=&dxc

defaultCompiler=nsc_release_upstream
supportsBinary=true
supportsBinaryObject=true
compilerType=nsc-spirv
needsMulti=false
supportsLibraryCodeFilter=true
disassemblerPath=@SPIRV_DIS_EXE@
demangler=

group.dxc.compilers=@NSC_COMPILERS@
group.dxc.includeFlag=-I
group.dxc.versionFlag=--version
group.dxc.groupName=NSC compilers
]=]
)

if(NSC_COMPILERS)
	string(CONFIGURE "${FINAL_CONFIG_CONTENT}" FINAL_CONFIG_CONTENT @ONLY)
	string(APPEND FINAL_CONFIG_CONTENT "${CONFIG_CONTENT}")
else()
	message(FATAL_ERROR "No compilers, internal error!")
endif()

file(WRITE "${OUTPUT_CONFIG_FILE}" "${FINAL_CONFIG_CONTENT}")
message(STATUS "Compiler Explorer configuration written to: ${OUTPUT_CONFIG_FILE}")
