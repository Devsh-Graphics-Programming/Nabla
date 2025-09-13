# How to update the DXC module

## FIRST TIME SETUP: Make sure you have the correct remotes!

The following submodules as initialized on your system should have `origin` pointing at the `git@github.com:Devsh-Graphics-Programming` fork.

Then a remote `Khronos` for the SPIR-V submodules, and `Microsoft` for the DXC.

If they don't then you can correct that with `git remote add [RemoteName] git@github.com:[Organisation]/[Repo-Name].git` and `git remote remove [RemoteName]`

## IF YOU GET `There is no tracking information for the current branch`

just make the branch track the origin
```
git branch --set-upstream-to=origin/[BranchName] [BranchName]
```

## Its Basically an Depth First Search with a Prologue and Epilogue

```
checkout correct branch HEAD
[optional] Merge latest stuff from original project
recurse()
commit
push
```

### First make sure you're on some Nabla branch Head

Just `git pull` and make sure you're tracking a branch.

### Make sure DXC is tracked and pointing at HEAD

```
cd ./3rdparty/dxc/dxc
git fetch
git checkout devshFixes
git pull
```

### SPIR-V Headers: track & get latest head, merge latest upstream, commit and push

```
cd ./external/SPIRV-Headers
git fetch
git checkout header_4_hlsl
git pull
git pull Khronos main
git commit -m "latest Khronos `main` merge"
git push
```

### SPIR-V Tools: track & get latest head, merge latest upstream, commit and push

```
cd ../SPIRV-Tools
git fetch
git checkout main
git pull
git pull Khronos main
git commit -m "latest Khronos `main` merge"
git push
```

### Back to DXC and commit the submodule pointer changes

```
cd ..
git add .
git commit -m "Update SPIR-V Headers and Tools"
cd ..
[optional] git pull Microsoft main
[optional] git commit -m "merge upstream from Microsoft"
git push
```

The reason the upstream from Microsoft is optional is because it might break our Clang hacks like:
- abuse of `__decltype`
- `inout` working like a reference, but only sometimes
- abuse of reference type

### Finally commit the change of DXC commit pointer to Nabla

```
cd ..
git add dxc
git commit -m "Updated DXC"
git push
```
