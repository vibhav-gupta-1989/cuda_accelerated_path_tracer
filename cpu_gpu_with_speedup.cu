// nvcc -O3 -arch=sm_60 cuda_tracer.cu -o tracer

#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <cmath>

#include <chrono>
#include <random>

#define WIDTH 800
#define HEIGHT 400
#define SAMPLES 100
#define MAX_DEPTH 5

std::mt19937 rng(42);
std::uniform_real_distribution<float> dist(0.0f, 1.0f);

inline float randf() {
    return dist(rng);
}

// ----------------------------------
// Vec3
// ----------------------------------
struct Vec3 {
    float x,y,z;

    __host__ __device__ Vec3() {}
    __host__ __device__ Vec3(float a,float b,float c):x(a),y(b),z(c){}

    __host__ __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x+b.x,y+b.y,z+b.z); }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x-b.x,y-b.y,z-b.z); }
    __host__ __device__ Vec3 operator-() const { return Vec3(-x,-y,-z); }

    __host__ __device__ Vec3 operator*(float t) const { return Vec3(x*t,y*t,z*t); }
    __host__ __device__ Vec3 operator*(const Vec3& b) const { return Vec3(x*b.x,y*b.y,z*b.z); }

    __host__ __device__ Vec3 operator/(float t) const { return *this*(1.0f/t); }

    __host__ __device__ Vec3& operator+=(const Vec3& b){
        x+=b.x; y+=b.y; z+=b.z; return *this;
    }
};

__host__ __device__
inline Vec3 operator*(float t, const Vec3& v) {
    return Vec3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__ inline float dot(const Vec3& a,const Vec3& b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline Vec3 normalize(Vec3 v){
    return v / sqrtf(dot(v,v));
}

// ----------------------------------
// Ray
// ----------------------------------
struct Ray {
    Vec3 orig, dir;
    __host__ __device__ Vec3 at(float t){ return orig + dir*t; }
};

// ----------------------------------
// Sphere
// ----------------------------------
struct Sphere {
    Vec3 center;
    float radius;
    int mat; // 0 diffuse, 1 metal, 2 glass
    Vec3 albedo;
    float fuzz;
    float ref_idx;
};

Vec3 random_in_unit_sphere_cpu(){
    while(true){
        Vec3 p(randf(), randf(), randf());
        p = p*2.0f - Vec3(1,1,1);
        if(dot(p,p) < 1.0f) return p;
    }
}

__host__ __device__ bool hit_sphere(Sphere s, Ray r, float tmin, float tmax, float& t, Vec3& normal){
    Vec3 oc = r.orig - s.center;
    float a = dot(r.dir,r.dir);
    float b = dot(oc,r.dir);
    float c = dot(oc,oc) - s.radius*s.radius;
    float disc = b*b - a*c;

    if(disc > 0){
        float temp = (-b - sqrtf(disc))/a;
        if(temp < tmax && temp > tmin){
            t = temp;
            normal = (r.at(t) - s.center)/s.radius;
            return true;
        }
    }
    return false;
}





// ----------------------------------
// Random helpers
// ----------------------------------
__device__ Vec3 random_in_unit_sphere(curandState* state){
    while(true){
        Vec3 p(
            curand_uniform(state),
            curand_uniform(state),
            curand_uniform(state)
        );
        p = p*2.0f - Vec3(1,1,1);
        if(dot(p,p) < 1.0f) return p;
    }
}

// ----------------------------------
// Material helpers
// ----------------------------------
__host__ __device__ Vec3 reflect(Vec3 v, Vec3 n){
    return v - 2*dot(v,n)*n;
}

__host__ __device__ Vec3 refract(Vec3 uv, Vec3 n, float etai_over_etat){
    float cos_theta = fminf(dot(-uv,n),1.0f);
    Vec3 r_out_perp = etai_over_etat*(uv + cos_theta*n);
    Vec3 r_out_parallel = -sqrtf(fabs(1.0f-dot(r_out_perp,r_out_perp)))*n;
    return r_out_perp + r_out_parallel;
}

__host__ __device__ float schlick(float cosine, float ref_idx){
    float r0 = (1-ref_idx)/(1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*powf((1-cosine),5);
}

// ----------------------------------
// Trace (iterative)
// ----------------------------------
__device__ Vec3 trace(Ray r, Sphere* spheres, int n, curandState* state){

    Vec3 throughput(1,1,1);

    for(int depth=0; depth<MAX_DEPTH; depth++){

        float closest = 1e20;
        int hit_id = -1;
        Vec3 normal;

        for(int i=0;i<n;i++){
            float t;
            Vec3 nrm;
            if(hit_sphere(spheres[i], r, 0.001f, closest, t, nrm)){
                closest = t;
                hit_id = i;
                normal = nrm;
            }
        }

        if(hit_id == -1){
            Vec3 unit = normalize(r.dir);
            float t = 0.5f*(unit.y + 1.0f);
            Vec3 sky = (1.0f-t)*Vec3(1,1,1) + t*Vec3(0.5,0.7,1.0);
            return throughput * sky;
        }

        Sphere s = spheres[hit_id];
        Vec3 hit = r.at(closest);

        if(s.mat == 0){
            Vec3 target = hit + normal + random_in_unit_sphere(state);
            r = {hit, normalize(target-hit)};
            throughput = throughput * s.albedo;
        }
        else if(s.mat == 1){
            Vec3 reflected = reflect(normalize(r.dir), normal);
            r = {hit, normalize(reflected + s.fuzz*random_in_unit_sphere(state))};
            throughput = throughput * s.albedo;
        }
        else{
            float etai = dot(r.dir,normal) > 0 ? s.ref_idx : 1.0f/s.ref_idx;

            Vec3 unit = normalize(r.dir);
            float cos_theta = fminf(dot(-unit,normal),1.0f);

            float reflect_prob = schlick(cos_theta, s.ref_idx);

            if(curand_uniform(state) < reflect_prob)
                r = {hit, reflect(unit,normal)};
            else
                r = {hit, refract(unit,normal,etai)};
        }
    }

    return Vec3(0,0,0);
}

// ----------------------------------
// Kernel
// ----------------------------------
__global__ void render(Vec3* fb, Sphere* spheres, int ns, curandState* rand_state){

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x>=WIDTH || y>=HEIGHT) return;

    int idx = y*WIDTH + x;
    curandState local = rand_state[idx];

    Vec3 col(0,0,0);

    for(int s=0;s<SAMPLES;s++){
        float u = (x + curand_uniform(&local)) / WIDTH;
        float v = (y + curand_uniform(&local)) / HEIGHT;

        Vec3 origin(0,2,6);
        Vec3 lookat(0,0,0);
        Vec3 vup(0,1,0);

        float focus_dist = 10.0f;
        float viewport_height = 2.0f;
        float viewport_width = 2.0f * float(WIDTH) / HEIGHT;

        // Camera basis
        Vec3 w = normalize(origin - lookat);
        Vec3 u_vec = normalize(Vec3(
            vup.y * w.z - vup.z * w.y,
            vup.z * w.x - vup.x * w.z,
            vup.x * w.y - vup.y * w.x
        ));
        Vec3 v_vec = Vec3(
            w.y * u_vec.z - w.z * u_vec.y,
            w.z * u_vec.x - w.x * u_vec.z,
            w.x * u_vec.y - w.y * u_vec.x
        );

        // Viewport
        Vec3 horizontal = focus_dist * viewport_width * u_vec;
        Vec3 vertical   = focus_dist * viewport_height * v_vec;
        Vec3 lower_left = origin - horizontal/2 - vertical/2 - focus_dist*w;

        // Ray direction
        Vec3 dir = normalize(lower_left + u*horizontal + v*vertical - origin);

        Ray r = {origin, dir};
        col += trace(r, spheres, ns, &local);
    }

    col = col / float(SAMPLES);
    col = Vec3(sqrt(col.x), sqrt(col.y), sqrt(col.z));

    fb[idx] = col;
    rand_state[idx] = local;
}

// ----------------------------------
// RNG init
// ----------------------------------
__global__ void init_rand(curandState* rand_state){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if(id < WIDTH*HEIGHT)
        curand_init(1984, id, 0, &rand_state[id]);
}

Vec3 trace_cpu(Ray r, Sphere* spheres, int n){

    Vec3 throughput(1,1,1);

    for(int depth=0; depth<MAX_DEPTH; depth++){

        float closest = 1e20;
        int hit_id = -1;
        Vec3 normal;

        for(int i=0;i<n;i++){
            float t;
            Vec3 nrm;
            if(hit_sphere(spheres[i], r, 0.001f, closest, t, nrm)){
                closest = t;
                hit_id = i;
                normal = nrm;
            }
        }

        if(hit_id == -1){
            Vec3 unit = normalize(r.dir);
            float t = 0.5f*(unit.y + 1.0f);
            Vec3 sky = (1.0f-t)*Vec3(1,1,1) + t*Vec3(0.5,0.7,1.0);
            return throughput * sky;
        }

        Sphere s = spheres[hit_id];
        Vec3 hit = r.at(closest);

        if(s.mat == 0){
            Vec3 target = hit + normal + random_in_unit_sphere_cpu();
            r = {hit, normalize(target-hit)};
            throughput = throughput * s.albedo;
        }
        else if(s.mat == 1){
            Vec3 reflected = reflect(normalize(r.dir), normal);
            r = {hit, normalize(reflected + s.fuzz*random_in_unit_sphere_cpu())};
            throughput = throughput * s.albedo;
        }
        else{
            float etai = dot(r.dir,normal) > 0 ? s.ref_idx : 1.0f/s.ref_idx;

            Vec3 unit = normalize(r.dir);
            float cos_theta = fminf(dot(-unit,normal),1.0f);

            float reflect_prob = schlick(cos_theta, s.ref_idx);

            if(randf() < reflect_prob)
                r = {hit, reflect(unit,normal)};
            else
                r = {hit, refract(unit,normal,etai)};
        }
    }

    return Vec3(0,0,0);
}

void render_cpu(Vec3* fb, Sphere* spheres, int N){

    Vec3 origin(0,2,6);
    Vec3 lookat(0,0,0);
    Vec3 vup(0,1,0);

    float focus_dist = 10.0f;
    float viewport_height = 2.0f;
    float viewport_width = 2.0f * float(WIDTH) / HEIGHT;

    Vec3 w = normalize(origin - lookat);
    Vec3 u_vec = normalize(Vec3(
        vup.y * w.z - vup.z * w.y,
        vup.z * w.x - vup.x * w.z,
        vup.x * w.y - vup.y * w.x
    ));
    Vec3 v_vec = Vec3(
        w.y * u_vec.z - w.z * u_vec.y,
        w.z * u_vec.x - w.x * u_vec.z,
        w.x * u_vec.y - w.y * u_vec.x
    );

    Vec3 horizontal = focus_dist * viewport_width * u_vec;
    Vec3 vertical   = focus_dist * viewport_height * v_vec;
    Vec3 lower_left = origin - horizontal/2 - vertical/2 - focus_dist*w;

    for(int j=0;j<HEIGHT;j++){
        for(int i=0;i<WIDTH;i++){

            Vec3 col(0,0,0);

            for(int s=0;s<SAMPLES;s++){
                float u = (i + randf()) / WIDTH;
                float v = (j + randf()) / HEIGHT;

                Vec3 dir = normalize(lower_left + u*horizontal + v*vertical - origin);
                Ray r = {origin, dir};

                col += trace_cpu(r, spheres, N);
            }

            col = col / float(SAMPLES);
            col = Vec3(sqrt(col.x), sqrt(col.y), sqrt(col.z));

            fb[j*WIDTH+i] = col;
        }
    }
}

// ----------------------------------
// MAIN
// ----------------------------------
int main(){

    Vec3* fb;
    cudaMallocManaged(&fb, WIDTH*HEIGHT*sizeof(Vec3));

    curandState* d_rand;
    cudaMalloc(&d_rand, WIDTH*HEIGHT*sizeof(curandState));

    init_rand<<<(WIDTH*HEIGHT+255)/256,256>>>(d_rand);
    cudaDeviceSynchronize();

    // Scene
    const int N = 50;
    Sphere* spheres;
    cudaMallocManaged((void**)&spheres, N*sizeof(Sphere));

    // Ground
    spheres[0] = {Vec3(0,-1000,0),1000,0,Vec3(0.5,0.5,0.5),0,0};

    // Main spheres
    spheres[1] = {Vec3(0,2,0),2,2,Vec3(1,1,1),0,1.1};   // glass
    spheres[2] = {Vec3(-4,2,0),2,0,Vec3(0.4,0.2,0.1),0,0}; // diffuse
    spheres[3] = {Vec3(4,2,0),2,1,Vec3(0.7,0.6,0.5),0.2,0}; // metal

    // Random small spheres
    int idx = 4;

    for(int a = -5; a < 5; a++){
        for(int b = -5; b < 5; b++){

            if(idx >= N) break;

            float choose_mat = randf();

            Vec3 center(
                a + 0.9f*randf(),
                0.2f,
                b + 0.9f*randf()
            );

            // Avoid overlap with main sphere
            if(dot(center - Vec3(0,2,0), center - Vec3(0,2,0)) < 4.0f)
                continue;

            if(choose_mat < 0.6f){
                // Diffuse
                spheres[idx++] = {
                    center,
                    0.2f,
                    0,
                    Vec3(randf()*randf(), randf()*randf(), randf()*randf()),
                    0,
                    0
                };
            }
            else if(choose_mat < 0.85f){
                // Metal
                spheres[idx++] = {
                    center,
                    0.2f,
                    1,
                    Vec3(0.5f*(1+randf()), 0.5f*(1+randf()), 0.5f*(1+randf())),
                    0.5f*randf(),
                    0
                };
            }
            else{
                // Glass
                spheres[idx++] = {
                    center,
                    0.2f,
                    2,
                    Vec3(1,1,1),
                    0,
                    1.5f
                };
            }
        }
    }

    cudaDeviceSynchronize();

    Vec3* fb_cpu = new Vec3[WIDTH*HEIGHT];

    // ---------------- CPU TIMING ----------------
    auto cpu_start = std::chrono::high_resolution_clock::now();

    render_cpu(fb_cpu, spheres, N);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(cpu_end - cpu_start).count();

    std::cout << "CPU Time: " << cpu_time << " seconds\n";


    // ---------------- GPU TIMING ----------------
    cudaDeviceSynchronize();
    auto gpu_start = std::chrono::high_resolution_clock::now();

    dim3 blocks((WIDTH+15)/16,(HEIGHT+15)/16);
    dim3 threads(16,16);

    render<<<blocks,threads>>>(fb, spheres, N, d_rand);
    cudaDeviceSynchronize();

    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(gpu_end - gpu_start).count();

    std::cout << "GPU Time: " << gpu_time << " seconds\n";


    // ---------------- SPEEDUP ----------------
    std::cout << "Speedup (CPU/GPU): " << cpu_time / gpu_time << "x\n";

    
    // Output PPM
    std::ofstream out("image.ppm");
    out<<"P3\n"<<WIDTH<<" "<<HEIGHT<<"\n255\n";

    for(int j=HEIGHT-1;j>=0;j--){
        for(int i=0;i<WIDTH;i++){
            int idx=j*WIDTH+i;
            int ir=int(255.99*fb[idx].x);
            int ig=int(255.99*fb[idx].y);
            int ib=int(255.99*fb[idx].z);
            out<<ir<<" "<<ig<<" "<<ib<<"\n";
        }
    }

    cudaFree(fb);
    cudaFree(spheres);
    cudaFree(d_rand);
}