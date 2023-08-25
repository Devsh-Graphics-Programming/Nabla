# How I generated the profiles
```
$ pip install jsonschema
```

```
$ cd ${VULKAN_SDK}/share/vulkan/registry
$ python gen_profiles_file.py --registry "${VULKAN_HEADERS}/registry/vk.xml" --input "PROFILES_DIR" -o "PROFILES_DIR/../profiles.json" --profile-desc "DESCRIPTION" --profile-label "NAME"
```

Note that this python script won't recurse, only profiles directly inside the directory will be intersected.

Profiles resulting from profiles can be further intersected.

## Platform notes

Android not taken into consideration yet.

Profiles missing for A13, A14 and M1 GPUs under iOS because no reports for Vulkan API version >=1.2.250

Also had to hand-edit the resulting `apple.json` profile to correct for that fact that MoltenVK only supports descriptor indexing properly with  `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=2`

## Working around bugs

You need to manually erase any fields dealing with Descriptor Buffer because of: https://github.com/SaschaWillems/VulkanCapsViewer/issues/188

There's another one with format merging: https://github.com/KhronosGroup/Vulkan-Profiles/issues/485