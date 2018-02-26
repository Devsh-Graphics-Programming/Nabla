#version 330 core

in vec4 Color; //per vertex output color, will be interpolated across the triangle
in vec3 Normal;

layout(location = 0) out vec4 pixelColor;

void main()
{
    vec3 normal = normalize(Normal);

    float ambient = 0.5;
    float diffuse = 0.5;
    float cos_theta_term = max(dot(normal,vec3(1.0,1.0,1.0)),0.0);

    float fresnel = 0.0; //not going to implement yet, not important
    float specular = 0.0;///pow(max(dot(halfVector,normal),0.0),shininess);

    pixelColor = Color*(ambient+mix(diffuse,specular,fresnel)*cos_theta_term/3.14159);
}
