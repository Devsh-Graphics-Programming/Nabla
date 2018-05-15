#version 430 core

in vec4 Color; //per vertex output color, will be interpolated across the triangle
flat in vec3 Normal;

layout(location = 0) out vec4 pixelColor;

void main()
{
    float ambient = 0.2;
    float diffuse = 0.8;
    float cos_theta_term = max(dot(Normal,vec3(1.0,1.0,1.0)),0.0);

    float fresnel = 0.0; //not going to implement yet, not important
    float specular = 0.0;///pow(max(dot(halfVector,Normal),0.0),shininess);

    const float sunPower = 3.14156*0.5;

    pixelColor = Color*sunPower*(ambient+mix(diffuse,specular,fresnel)*cos_theta_term/3.14159);
}
