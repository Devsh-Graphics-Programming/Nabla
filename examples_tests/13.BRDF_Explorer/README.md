# BRDF Explorer

## How to use

### Material Parameters window
* Use sliders to set emissive and albedo (base) colors. You can also use color picker instead by clicking on "Choose color"
* Then below you can use two roughness sliders to adjust roughness itself and anisotropy coefficient (range 0 to 1). If set to 0, Lambertian diffuse model is used. Otherwise it's always Oren-Nayar.
* Selecting Isotropic checkbox makes sure, that anisotropy coefficient is 0 and disables its slider
* Further below you can see sliders. They're for adjusting index of refraction (which tells how much light slows down in material).
Index of refraction is complex (the imaginary is often called "extinction coefficient"), that's why there are 6 sliders: real and imaginary 
part for wavelengths corresponding to red, green and blue colors.
* Next is metallic. Metallic is enabled only if index of refraction is based on data from reflectance texture. Otherwise it is disabled and you can treat it as it's constant 0.
Metallic parameter tells you how much of conductor the material is. It's best if it's set to either 0 or 1, but you can experiment with other values to get desired look.
Fully metallic materials does not present diffuse light.
* Next thing you can set bump/height map and choose height factor from slider. Then from bump map and height factor derivative map is generated and actually in shaders.
* At the very bottom of this window, you can set texture AO. Make sure to select Enabled checkbox for it to be used.
* As you might have notices, there are for dropdown menus for albedo, roughness, index of refraction and metallic parameter.
Using those you can choose whether you want aforementioned values to be constant or gathered from textures. You can set any of "Texture 0..3" which corresponds to one of textures you can see on the right, in Texture Preview window.
* For index of refraction reflectance texture is expected. Then real part of index of refraction is calculated from F0 (which is interpolation of F0-for-dielectrics (calculated from reflectance) and base color with metallic parameter being the value for interpolation).

### Light Parameters window
* Here you can set light color (in similar manner as it is done with emissive and albedo), its position and intensity
* There's also Animation checkbox. By selecting it, you can make light circle around the mesh. This obviously means, that position set in text-fields above is irrelevant.

### Texture Preview
* Textures are loaded along with models (for more see *Load Model button* section below), but you can also load them separately. Click on any slot you want the new texture to appear.

### Load Model button
* You can load as many models as you want. Also note that, once loaded model will not have to be loaded again - you have to point the file it comes from again, but it will show up immidiately.
* If any textures come with loaded they're also loaded (to 4 slots in Texture Preview window), but only if slot is empty - loading model does not cause overriding choice of textures.

Purely red pixels mean physically impossible values.
