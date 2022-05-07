//
// Created by roboevt on 5/6/22.
//

#include "RayTracer.h"
#include "Tracer.cuh"

namespace {
    CudaTracer tracer;
    CudaColor backgroundColor(.5f,.5f,.5f);
}

Color::Color(float r, float g, float b) : r(r), g(g), b(b) {}

Sphere::Sphere(float x, float y, float z, float r) : x(x), y(y), z(z), r(r) {}

CudaSphere toCudaSphere(Sphere sphere) {
    CudaSphere cudaSphere(vec3(sphere.x,sphere.y,sphere.z),sphere.r);
    cudaSphere.material.color = CudaColor(sphere.color.r, sphere.color.g, sphere.color.b);
    return cudaSphere;
}

Window::Window(int width, int height, std::string name) : width(width), height(height), name(name) {
    glfwInit();
    window = glfwCreateWindow(this->width, this->height, name.c_str(), NULL, NULL);
    glfwMakeContextCurrent(window);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glGenBuffersARB(1, &openGLPixelBuffer);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, openGLPixelBuffer);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, this->width * this->height * sizeof(float4), 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

GLuint Window::getGLBuffer() {return openGLPixelBuffer;}

int Window::displayFrame() {
    if (glfwWindowShouldClose(window)) return 1;
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(-1, -1);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, openGLPixelBuffer);
    glDrawPixels(width, height, GL_RGBA, GL_FLOAT, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glfwSwapBuffers(window);
    glfwPollEvents();
    return 0;
}
Window::~Window() {
    glDeleteBuffersARB(1, &openGLPixelBuffer);
}

void RayTracer::init(GLuint openGLPixelBuffer) {
    size_t numSpheres = spheres.size();
    CudaSphere* cudaSpheres;
    cudaSpheres = new CudaSphere[numSpheres];
    for(int i = 0; i < numSpheres; i++) {
        cudaSpheres[i] = toCudaSphere(spheres[i]);
    }
    World world(numSpheres, cudaSpheres, &backgroundColor);
    tracer.setWorld(world);
    Camera camera(vec3(0,.5f,0), 1.0f, 1920,1080);
    tracer = CudaTracer(world, camera, openGLPixelBuffer);
    tracer.setSamples(10);  // TODO parameter
}

void RayTracer::addSphere(Sphere sphere) {
    spheres.push_back(sphere);
}

void RayTracer::renderFrame(bool clearFrame) {
    tracer.renderFrame(clearFrame);
}

