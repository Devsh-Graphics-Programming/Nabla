#ifndef _NBL_SYSTEM_RUNTIME_MODULE_LOOKUP_H_INCLUDED_
#define _NBL_SYSTEM_RUNTIME_MODULE_LOOKUP_H_INCLUDED_

#include "nbl/system/ModuleLookupUtils.h"

namespace nbl::system
{
struct RuntimeModuleLookup final
{
    struct LookupPaths
    {
        system::path install;
        system::path build;
    };

    struct Module
    {
        LookupPaths paths;
        std::string_view name = "";
        std::string_view buildOutputDir = "";
        std::string_view buildDllPath = "";
        std::string_view installOverrideRel = "";
        std::string_view installBuildFallbackRel = "";
        std::string_view runtimeAbsKey = "";
    };

    bool sharedBuild = false;
    bool relocatablePackage = false;
    Module nabla;
    Module dxc;

    RuntimeModuleLookup()
    {
        dxc.name = "dxcompiler";
        #if defined(_NBL_SHARED_BUILD_)
        sharedBuild = true;
        nabla.name = _NABLA_DLL_NAME_;
        #endif
        #if defined(NBL_RELOCATABLE_PACKAGE)
        relocatablePackage = true;
        #endif
        #if defined(_NABLA_OUTPUT_DIR_)
        nabla.buildOutputDir = _NABLA_OUTPUT_DIR_;
        #endif
        #if defined(_DXC_DLL_)
        dxc.buildDllPath = _DXC_DLL_;
        #endif
        #if defined(NBL_CPACK_PACKAGE_NABLA_DLL_DIR)
        nabla.installOverrideRel = NBL_CPACK_PACKAGE_NABLA_DLL_DIR;
        #endif
        #if defined(NBL_CPACK_PACKAGE_DXC_DLL_DIR)
        dxc.installOverrideRel = NBL_CPACK_PACKAGE_DXC_DLL_DIR;
        #endif
        #if defined(NBL_CPACK_PACKAGE_NABLA_DLL_DIR_BUILD_FALLBACK)
        nabla.installBuildFallbackRel = NBL_CPACK_PACKAGE_NABLA_DLL_DIR_BUILD_FALLBACK;
        #endif
        #if defined(NBL_CPACK_PACKAGE_DXC_DLL_DIR_BUILD_FALLBACK)
        dxc.installBuildFallbackRel = NBL_CPACK_PACKAGE_DXC_DLL_DIR_BUILD_FALLBACK;
        #endif
        #if defined(NBL_CPACK_PACKAGE_NABLA_DLL_DIR_ABS_KEY)
        nabla.runtimeAbsKey = NBL_CPACK_PACKAGE_NABLA_DLL_DIR_ABS_KEY;
        #endif
        #if defined(NBL_CPACK_PACKAGE_DXC_DLL_DIR_ABS_KEY)
        dxc.runtimeAbsKey = NBL_CPACK_PACKAGE_DXC_DLL_DIR_ABS_KEY;
        #endif

        applyBuildInterfacePaths();
    }

    inline void applyInstallOverrides(const system::path& exeDirectory)
    {
        if (hasInstallOverride(nabla))
            nabla.paths.install = absoluteFromExe(exeDirectory, nabla.installOverrideRel);
        if (hasInstallOverride(dxc))
            dxc.paths.install = absoluteFromExe(exeDirectory, dxc.installOverrideRel);
    }

    inline bool chooseInstallLookupMode(const system::path& exeDirectory)
    {
        if (relocatablePackage)
        {
            if (!hasUsableInstallPaths())
            {
                if (!tryResolveInstallPathsFromPackageLayout(exeDirectory))
                    tryResolveInstallPathsFromBuildFallbackHints(exeDirectory);
            }
            return true;
        }
        if (hasUsableInstallPaths())
            return true;
        if (tryResolveInstallPathsFromPackageLayout(exeDirectory))
            return true;
        return tryResolveInstallPathsFromBuildFallbackHints(exeDirectory);
    }

    inline void finalizeInstallLookups(bool useInstallLookups)
    {
        if (!useInstallLookups)
            return;
        #if defined(_NBL_PLATFORM_WINDOWS_) && defined(_NBL_SHARED_BUILD_)
        if (nabla.paths.install.empty())
            nabla.paths.install = loadedModuleDirectory(nabla.name);
        #endif
        resolveDxcInstallPathFromLoadedNabla(useInstallLookups);
    }

    private:
    static inline bool hasInstallOverride(const Module& module)
    {
        return !module.installOverrideRel.empty();
    }

    static inline bool hasRuntimeAbsKey(const Module& module)
    {
        return !module.runtimeAbsKey.empty();
    }

    inline void applyBuildInterfacePaths()
    {
        if (sharedBuild && !nabla.buildOutputDir.empty())
            nabla.paths.build = system::path(nabla.buildOutputDir);
        if (!dxc.buildDllPath.empty())
            dxc.paths.build = system::path(dxc.buildDllPath).parent_path();
    }

    static inline system::path absoluteFromExe(const system::path& exeDirectory, std::string_view relativePath)
    {
        if (relativePath.empty() || exeDirectory.empty())
            return system::path("");

        const auto relPath = system::path(relativePath);
        if (relPath.is_absolute())
            return system::path("");

        return std::filesystem::absolute(exeDirectory / relPath);
    }

    inline bool hasUsableInstallPaths() const
    {
        if (!moduleExistsInDirectory(dxc.paths.install, dxc.name))
            return false;
        return !sharedBuild || moduleExistsInDirectory(nabla.paths.install, nabla.name);
    }

    inline bool tryResolveInstallPathsFromPrefix(const system::path& candidatePrefix)
    {
        if (candidatePrefix.empty())
            return false;
        if (!hasRuntimeAbsKey(nabla) && !hasRuntimeAbsKey(dxc))
            return false;

        Module candidateNabla = nabla;
        Module candidateDxc = dxc;

        if (hasRuntimeAbsKey(nabla))
            candidateNabla.paths.install = std::filesystem::absolute(candidatePrefix / system::path(nabla.runtimeAbsKey));
        if (hasRuntimeAbsKey(dxc))
            candidateDxc.paths.install = std::filesystem::absolute(candidatePrefix / system::path(dxc.runtimeAbsKey));

        if (!moduleExistsInDirectory(candidateDxc.paths.install, candidateDxc.name))
            return false;
        if (sharedBuild && !moduleExistsInDirectory(candidateNabla.paths.install, candidateNabla.name))
            return false;

        nabla.paths.install = candidateNabla.paths.install;
        dxc.paths.install = candidateDxc.paths.install;
        return true;
    }

    inline bool tryResolveInstallPathsFromPackageLayout(const system::path& lookupStartDirectory)
    {
        if (lookupStartDirectory.empty())
            return false;
        if (!hasRuntimeAbsKey(nabla) && !hasRuntimeAbsKey(dxc))
            return false;

        auto candidatePrefix = std::filesystem::absolute(lookupStartDirectory);
        while (!candidatePrefix.empty())
        {
            if (tryResolveInstallPathsFromPrefix(candidatePrefix))
                return true;

            const auto parent = candidatePrefix.parent_path();
            if (parent == candidatePrefix)
                break;
            candidatePrefix = parent;
        }
        return false;
    }

    inline bool tryResolveInstallPathsFromBuildFallbackHints(const system::path& exeDirectory)
    {
        Module candidateNabla = nabla;
        Module candidateDxc = dxc;
        candidateNabla.paths.install = system::path("");
        candidateDxc.paths.install = system::path("");

        if (!candidateNabla.installBuildFallbackRel.empty())
            candidateNabla.paths.install = absoluteFromExe(exeDirectory, candidateNabla.installBuildFallbackRel);
        if (!candidateDxc.installBuildFallbackRel.empty())
            candidateDxc.paths.install = absoluteFromExe(exeDirectory, candidateDxc.installBuildFallbackRel);

        if (candidateDxc.paths.install.empty() && !candidateNabla.paths.install.empty() && hasRuntimeAbsKey(nabla) && hasRuntimeAbsKey(dxc))
        {
            const auto dxcRelToNabla = system::path(dxc.runtimeAbsKey).lexically_relative(system::path(nabla.runtimeAbsKey));
            if (!dxcRelToNabla.empty() && dxcRelToNabla != system::path("."))
                candidateDxc.paths.install = std::filesystem::absolute(candidateNabla.paths.install / dxcRelToNabla);
        }

        if (!moduleExistsInDirectory(candidateDxc.paths.install, candidateDxc.name))
            return false;
        if (sharedBuild && !moduleExistsInDirectory(candidateNabla.paths.install, candidateNabla.name))
            return false;

        nabla.paths.install = candidateNabla.paths.install;
        dxc.paths.install = candidateDxc.paths.install;
        return true;
    }

    #if defined(_NBL_PLATFORM_WINDOWS_)
    inline void resolveDxcInstallPathFromLoadedNabla(bool useInstallLookups)
    {
        if (!useInstallLookups || !dxc.paths.install.empty())
            return;
        if (!(sharedBuild && !nabla.runtimeAbsKey.empty() && !dxc.runtimeAbsKey.empty()))
            return;

        const auto nablaRuntimeDir = !nabla.paths.install.empty() ? nabla.paths.install : loadedModuleDirectory(nabla.name);
        if (nablaRuntimeDir.empty())
            return;

        const auto dxcRelToNabla = system::path(dxc.runtimeAbsKey).lexically_relative(system::path(nabla.runtimeAbsKey));
        if (!dxcRelToNabla.empty() && dxcRelToNabla != system::path("."))
            dxc.paths.install = std::filesystem::absolute(nablaRuntimeDir / dxcRelToNabla);
    }
    #else
    inline void resolveDxcInstallPathFromLoadedNabla(bool)
    {
    }
    #endif
};
}

#endif
