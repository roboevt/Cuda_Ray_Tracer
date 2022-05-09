//
// Created by roboevt on 5/6/22.
//

#include "RayTracer.h"
#include "Tracer.cuh"

namespace {
    CudaTracer tracer;
    CudaColor backgroundColor(.05f,.05f,.05f);
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

void Window::setKeyCallback(GLFWkeyfun callback) {
    glfwSetKeyCallback(window, callback);
}

Window::~Window() {
    glDeleteBuffersARB(1, &openGLPixelBuffer);
}

RayTracer::RayTracer(Window *window){
    RayTracer::window = window;
}

void RayTracer::addSphere(Sphere sphere) {spheres.push_back(sphere); }
void RayTracer::addSpheres(Sphere* spheres, int numSpheres) {
    std::copy(spheres, spheres + numSpheres, std::back_inserter(this->spheres));
}

void RayTracer::setSamples(int samples) {this->samples = samples; }

void RayTracer::renderFrame(bool clearFrame) {
    tracer.renderFrame(clearFrame);
}

void RayTracer::keyPressCallback(GLFWwindow *window, int key, int scancode,
                                 int action, int mods) {
    Camera cam = tracer.getCamera();
    if(action == GLFW_REPEAT) {  // TODO takes a bit to activate
        switch(key) {
            case GLFW_KEY_R:
                cam.zoom +=.1f;
                break;
            case GLFW_KEY_F:
                cam.zoom -=.1f;
                break;
            case GLFW_KEY_W:
                cam.origin.z+=.1f;
                break;
            case GLFW_KEY_A:
                cam.origin.x -= .1f;
                break;
            case GLFW_KEY_D:
                cam.origin.x +=.1f;
                break;
            case GLFW_KEY_S:
                cam.origin.z -=.1f;
                break;
            default:
                break;
        }
    }
    tracer.setCamera(Camera(cam.origin, cam.zoom, tracer.getWidth(), tracer.getHeight()));
}

void RayTracer::init() {
    size_t numSpheres = this->spheres.size();
    CudaSphere* cudaSpheres;
    cudaSpheres = new CudaSphere[numSpheres];
    for(int i = 0; i < numSpheres; i++) {
        cudaSpheres[i] = toCudaSphere(this->spheres[i]);
    }
    World world(numSpheres, cudaSpheres, &backgroundColor);
    tracer.setWorld(world);
    Camera cudaCam(vec3(0, 0.5f, 0), 1.0f, this->window->width, this->window->height);
    tracer = CudaTracer(world, cudaCam, this->window->getGLBuffer());
    tracer.setSamples(this->samples);
}