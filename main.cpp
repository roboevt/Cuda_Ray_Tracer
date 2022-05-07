#include "RayTracer.h"

constexpr int width = 1920;
constexpr int height = 1080;

int main(void)
{
    Window window(width, height, "Cuda Ray Tracer");

    Sphere sphere = Sphere(-1.0f, 1.0f, 8.0f, 1.0f);
    sphere.color = Color(1, 1, 1);
    Sphere sphere2 = Sphere(1.0f, 1.0f, 8.0f, 1.0f);
    sphere2.color = Color(.5f, .5f, .5f);
    Sphere sphere3 = Sphere(0.0f, -100.0f, 8.0f, 135.0f);
    sphere3.color = Color(1, 1, 1);
    Sphere sphere4 = Sphere(0.0f, 5.5f, 4.0f, 0.5f);
    sphere4.color = Color(60, 80, 90);
    Sphere sphere5 = Sphere(1.5f, 0.2f, 5.5f, 0.2f);
    sphere5.color = Color(40, 15, 10);

    RayTracer rayTracer;
    rayTracer.addSphere(sphere);
    rayTracer.addSphere(sphere2);
    rayTracer.addSphere(sphere3);
    rayTracer.addSphere(sphere4);
    rayTracer.addSphere(sphere5);
    rayTracer.init(window.getGLBuffer());
    printf("Done initialization\n");
    while(1) {
        rayTracer.renderFrame(false);
        if(window.displayFrame()) break;
    }

    return 0;
}