#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <cstdlib>

#define WIDTH 800
#define HEIGHT 400
#define scene_size 40
#define SAMPLES 5000
#define MAX_BOUNCES 5

////////////////////////////////////////////////////////////

struct Vec3{
    float x,y,z;

    __host__ __device__ Vec3(){}
    __host__ __device__ Vec3(float a,float b,float c):x(a),y(b),z(c){}

    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x+v.x,y+v.y,z+v.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x-v.x,y-v.y,z-v.z);
    }

    __host__ __device__ Vec3 operator*(float t) const {
        return Vec3(x*t,y*t,z*t);
    }
};

////////////////////////////////////////////////////////////

struct Ray{
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray(Vec3 o,Vec3 d):origin(o),direction(d){}
};

////////////////////////////////////////////////////////////

struct Sphere{
    Vec3 center;
    float radius;
    Vec3 color;
};

////////////////////////////////////////////////////////////
// RNG
////////////////////////////////////////////////////////////

__device__ float randf(unsigned int &seed)
{
    seed = 1664525u * seed + 1013904223u;
    return (seed & 0x00FFFFFF) / float(0x01000000);
}

float randf_cpu(unsigned int &seed)
{
    seed = 1664525u * seed + 1013904223u;
    return (seed & 0x00FFFFFF) / float(0x01000000);
}

////////////////////////////////////////////////////////////
// Random direction
////////////////////////////////////////////////////////////

__device__ Vec3 random_in_unit_sphere(unsigned int &seed)
{
    while(true)
    {
        Vec3 p(
            randf(seed)*2.0f - 1.0f,
            randf(seed)*2.0f - 1.0f,
            randf(seed)*2.0f - 1.0f
        );

        if(p.x*p.x + p.y*p.y + p.z*p.z < 1.0f)
            return p;
    }
}

Vec3 random_in_unit_sphere_cpu(unsigned int &seed)
{
    while(true)
    {
        Vec3 p(
            randf_cpu(seed)*2.0f - 1.0f,
            randf_cpu(seed)*2.0f - 1.0f,
            randf_cpu(seed)*2.0f - 1.0f
        );

        if(p.x*p.x + p.y*p.y + p.z*p.z < 1.0f)
            return p;
    }
}

////////////////////////////////////////////////////////////
// Sphere intersection
////////////////////////////////////////////////////////////

__device__
bool hit_sphere(Sphere s, Ray r, float &t, Vec3 &normal)
{
    Vec3 oc = r.origin - s.center;

    float a = r.direction.x*r.direction.x +
              r.direction.y*r.direction.y +
              r.direction.z*r.direction.z;

    float b = 2*(oc.x*r.direction.x +
                 oc.y*r.direction.y +
                 oc.z*r.direction.z);

    float c = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z -
              s.radius*s.radius;

    float disc = b*b - 4*a*c;

    if(disc > 0)
    {
        float temp = (-b - sqrtf(disc)) / (2*a);
        if(temp > 0.001f)
        {
            t = temp;
            Vec3 hit = r.origin + r.direction * t;
            normal = (hit - s.center);

            float len = sqrtf(normal.x*normal.x +
                              normal.y*normal.y +
                              normal.z*normal.z);

            normal = normal * (1.0f / len);
            return true;
        }
    }

    return false;
}

bool hit_sphere_cpu(Sphere s, Ray r, float &t, Vec3 &normal)
{
    Vec3 oc = r.origin - s.center;

    float a = r.direction.x*r.direction.x +
              r.direction.y*r.direction.y +
              r.direction.z*r.direction.z;

    float b = 2*(oc.x*r.direction.x +
                 oc.y*r.direction.y +
                 oc.z*r.direction.z);

    float c = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z -
              s.radius*s.radius;

    float disc = b*b - 4*a*c;

    if(disc > 0)
    {
        float temp = (-b - sqrtf(disc)) / (2*a);
        if(temp > 0.001f)
        {
            t = temp;
            Vec3 hit = r.origin + r.direction * t;
            normal = (hit - s.center);

            float len = sqrtf(normal.x*normal.x +
                              normal.y*normal.y +
                              normal.z*normal.z);

            normal = normal * (1.0f / len);
            return true;
        }
    }

    return false;
}

////////////////////////////////////////////////////////////
// PATH TRACING
////////////////////////////////////////////////////////////

__device__
Vec3 ray_color(Ray r, Sphere* spheres, unsigned int &seed)
{
    Vec3 throughput(1,1,1);

    for(int bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        float closest = 1e20;
        int hit_id = -1;
        Vec3 normal;

        for(int i = 0; i < scene_size; i++)
        {
            float t;
            Vec3 n;

            if(hit_sphere(spheres[i], r, t, n))
            {
                if(t < closest)
                {
                    closest = t;
                    hit_id = i;
                    normal = n;
                }
            }
        }

        if(hit_id != -1)
        {
            Sphere s = spheres[hit_id];

            if(s.color.x > 5.0f)
            {
                return Vec3(
                    throughput.x * s.color.x,
                    throughput.y * s.color.y,
                    throughput.z * s.color.z
                );
            }

            Vec3 hit_point = r.origin + r.direction * closest;

            Vec3 target = hit_point + normal + random_in_unit_sphere(seed);

            Vec3 offset = normal * 0.001f;
            r = Ray(hit_point + offset, target - hit_point);

            throughput = Vec3(
                throughput.x * s.color.x,
                throughput.y * s.color.y,
                throughput.z * s.color.z
            );
        }
        else
        {
            float t = 0.5f * (r.direction.y + 1.0f);

            Vec3 sky =
                Vec3(1,1,1)*(1.0f - t) +
                Vec3(0.5,0.7,1.0f)*t;

            return Vec3(
                throughput.x * sky.x,
                throughput.y * sky.y,
                throughput.z * sky.z
            );
        }
    }

    return Vec3(0,0,0);
}

Vec3 ray_color_cpu(Ray r, Sphere* spheres, unsigned int &seed)
{
    Vec3 throughput(1,1,1);

    for(int bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        float closest = 1e20;
        int hit_id = -1;
        Vec3 normal;

        for(int i = 0; i < scene_size; i++)
        {
            float t;
            Vec3 n;

            if(hit_sphere_cpu(spheres[i], r, t, n))
            {
                if(t < closest)
                {
                    closest = t;
                    hit_id = i;
                    normal = n;
                }
            }
        }

        if(hit_id != -1)
        {
            Sphere s = spheres[hit_id];

            if(s.color.x > 5.0f)
            {
                return Vec3(
                    throughput.x * s.color.x,
                    throughput.y * s.color.y,
                    throughput.z * s.color.z
                );
            }

            Vec3 hit_point = r.origin + r.direction * closest;

            Vec3 target = hit_point + normal + random_in_unit_sphere_cpu(seed);

            Vec3 offset = normal * 0.001f;
            r = Ray(hit_point + offset, target - hit_point);

            throughput = Vec3(
                throughput.x * s.color.x,
                throughput.y * s.color.y,
                throughput.z * s.color.z
            );
        }
        else
        {
            float t = 0.5f * (r.direction.y + 1.0f);

            Vec3 sky =
                Vec3(1,1,1)*(1.0f - t) +
                Vec3(0.5,0.7,1.0f)*t;

            return Vec3(
                throughput.x * sky.x,
                throughput.y * sky.y,
                throughput.z * sky.z
            );
        }
    }

    return Vec3(0,0,0);
}

////////////////////////////////////////////////////////////
// GPU kernel
////////////////////////////////////////////////////////////

__global__
void render(Vec3* fb, Sphere* spheres)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= WIDTH || y >= HEIGHT) return;

    int pixel = y*WIDTH + x;

    unsigned int seed = pixel;

    Vec3 col(0,0,0);

    for(int s = 0; s < SAMPLES; s++)
    {
        float u = (x + randf(seed)) / WIDTH;
        float v = (y + randf(seed)) / HEIGHT;

        Vec3 dir = Vec3(u*2-1, v*2-1, -1);
        float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
        dir = dir * (1.0f / len);

        Ray r(Vec3(0,0,0), dir);

        Vec3 c = ray_color(r, spheres, seed);

        col = col + c;
    }

    fb[pixel] = col * (1.0f / SAMPLES);
}

////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////

int main()
{
    int pixels = WIDTH * HEIGHT;

    Vec3* fb;
    cudaMallocManaged(&fb, pixels*sizeof(Vec3));

    Sphere* spheres;
    cudaMallocManaged(&spheres, scene_size*sizeof(Sphere));

    srand(42);

    int idx = 0;

    // Light
    spheres[idx++] = {Vec3(0,1.5,-3),0.25,Vec3(15,15,15)};
    // Floor
    spheres[idx++] = {Vec3(0,-1001,-3),1000,Vec3(0.8,0.8,0.8)};
    // Ceiling
    spheres[idx++] = {Vec3(0,1003,-3),1000,Vec3(0.8,0.8,0.8)};
    // Back wall
    spheres[idx++] = {Vec3(0,0,-1005),1000,Vec3(0.8,0.8,0.8)};
    // Left
    spheres[idx++] = {Vec3(-1003,0,-3),1000,Vec3(0.9,0.1,0.1)};
    // Right
    spheres[idx++] = {Vec3(1003,0,-3),1000,Vec3(0.1,0.9,0.1)};
    // Objects
    // Random small spheres
    
    unsigned int seed = 1337;

    for(int a = -3; a <= 3 && idx < scene_size; a++)
    {
        for(int b = -3; b <= 3 && idx < scene_size; b++)
        {
            float x = a + 0.4f * randf_cpu(seed);
            float z = -2.5f + b * 0.5f;

            float radius = 0.2f + 0.2f * randf_cpu(seed);

            Vec3 color(
                randf_cpu(seed),
                randf_cpu(seed),
                randf_cpu(seed)
            );

            spheres[idx++] = {Vec3(x, -0.8f + radius, z), radius, color};
        }
    }

    ////////////////////////////////////////////////////////
    // CPU Render
    ////////////////////////////////////////////////////////

    Vec3* fb_cpu = new Vec3[pixels];

    auto cpu_start = std::chrono::high_resolution_clock::now();

    for(int y=0;y<HEIGHT;y++)
    {
        for(int x=0;x<WIDTH;x++)
        {
            int pixel = y*WIDTH+x;
            unsigned int seed = pixel;

            Vec3 col(0,0,0);

            for(int s=0;s<SAMPLES;s++)
            {
                float u = (x + randf_cpu(seed)) / WIDTH;
                float v = (y + randf_cpu(seed)) / HEIGHT;

                Vec3 dir = Vec3(u*2-1,v*2-1,-1);
                float len = sqrtf(dir.x*dir.x+dir.y*dir.y+dir.z*dir.z);
                dir = dir*(1.0f/len);

                Ray r(Vec3(0,0,0),dir);
                col = col + ray_color_cpu(r,spheres,seed);
            }

            fb_cpu[pixel] = col*(1.0f/SAMPLES);
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(cpu_end-cpu_start).count();

    ////////////////////////////////////////////////////////
    // GPU Render
    ////////////////////////////////////////////////////////

    dim3 threads(16,16);
    dim3 blocks((WIDTH+15)/16,(HEIGHT+15)/16);

    auto gpu_start = std::chrono::high_resolution_clock::now();

    render<<<blocks,threads>>>(fb,spheres);
    cudaDeviceSynchronize();

    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(gpu_end-gpu_start).count();

    ////////////////////////////////////////////////////////
    // Speedup
    ////////////////////////////////////////////////////////

    std::cout<<"CPU Time: "<<cpu_time<<" sec\n";
    std::cout<<"GPU Time: "<<gpu_time<<" sec\n";
    std::cout<<"Speedup: "<<cpu_time/gpu_time<<"x\n";

    ////////////////////////////////////////////////////////
    // Output (GPU)
    ////////////////////////////////////////////////////////

    std::ofstream image("output.ppm");
    image<<"P3\n"<<WIDTH<<" "<<HEIGHT<<"\n255\n";

    for(int j=HEIGHT-1;j>=0;j--)
    {
        for(int i=0;i<WIDTH;i++)
        {
            int pixel=j*WIDTH+i;

            float r=sqrtf(fb[pixel].x);
            float g=sqrtf(fb[pixel].y);
            float b=sqrtf(fb[pixel].z);

            image<<(int)(255.99*r)<<" "
                 <<(int)(255.99*g)<<" "
                 <<(int)(255.99*b)<<"\n";
        }
    }

    image.close();

    cudaFree(fb);
    cudaFree(spheres);
    delete[] fb_cpu;
}