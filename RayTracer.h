//
// Created by roboevt on 5/6/22.
//

#ifndef RAYTRACER_RAYTRACER_H
#define RAYTRACER_RAYTRACER_H

#define GL_GLEXT_PROTOTYPES

#include <vector>
#include <string>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html
//https://github.com/phrb/intro-cuda/blob/master/src/cuda-samples/2_Graphics/bindlessTexture/bindlessTexture.cpp
class Window {
    GLFWwindow* window;
    GLuint openGLPixelBuffer;
    int width, height;
    std::string name;
public:
    Window(int width, int height, std::string name);
    GLuint getGLBuffer();
    int displayFrame();
    ~Window();
};

struct Color {
    float r, g, b;
    Color(float r, float g, float b);
    Color() = default;
};

struct Sphere {
    float x, y, z, r;
    Color color;
    Sphere(float x, float y, float z, float r);
};

class RayTracer {
    std::vector<Sphere> spheres;
public:
    void init(GLuint openGLPixelBuffer);
    void addSphere(Sphere sphere);
    void renderFrame(bool clearFrame);
};


#endif //RAYTRACER_RAYTRACER_H
