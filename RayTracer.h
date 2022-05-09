//
// Created by roboevt on 5/6/22.
//

#ifndef RAYTRACER_RAYTRACER_H
#define RAYTRACER_RAYTRACER_H

#define GL_GLEXT_PROTOTYPES

#include <vector>
#include <string>
#include <GLFW/glfw3.h>


//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html
//https://github.com/phrb/intro-cuda/blob/master/src/cuda-samples/2_Graphics/bindlessTexture/bindlessTexture.cpp
class Window {
    GLFWwindow* window;
    GLuint openGLPixelBuffer;
public:
    int width, height;
    std::string name;
    Window(int width, int height, std::string name);
    GLuint getGLBuffer();
    int displayFrame();
    void setKeyCallback(GLFWkeyfun callback);
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
    Window* window;
    std::vector<Sphere> spheres;
    int samples;
public:
    RayTracer(Window* window);
    void addSphere(Sphere sphere);
    void addSpheres(Sphere* spheres, int numSpheres);
    void setSamples(int samples);
    void renderFrame(bool clearFrame);
    static void keyPressCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    int getWidth() {return window->width;}
    int getHeight() {return window->height;}
    void init();
};


#endif //RAYTRACER_RAYTRACER_H
