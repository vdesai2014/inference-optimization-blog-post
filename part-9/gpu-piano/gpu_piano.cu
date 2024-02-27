#include <iostream>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <thread>
#include <pthread.h>
#include <cuda_runtime.h>
#include <SDL.h>
#include <unordered_map>

std::atomic<bool> keyIsPressed(false);
std::atomic<long> idleDuration(5); 

__global__ void memoryIntensiveKernel(float *data, int size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;
    for (int i = idx; i < size; i+=stride) {
        temp += data[i];
    }
    if (idx == 0) data[0] = temp;
}

void* listenForKeyPress(void* arg) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return (void*)-1;
    }

    SDL_Window* window = SDL_CreateWindow("Keyboard Input Listener",
                                          SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          640, 480, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return (void*)-1;
    }

    // Define key to idleDuration mapping
    std::unordered_map<SDL_Keycode, int> keyDurationMapping = {
        {SDLK_a, 158},
        {SDLK_s, 70},
        {SDLK_d, 38},
        {SDLK_j, 99},
        {SDLK_k, 112},
        {SDLK_l, 37},
    };

    bool quit = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            } else if (e.type == SDL_KEYDOWN) {
                keyIsPressed.store(true);
                auto it = keyDurationMapping.find(e.key.keysym.sym);
                if (it != keyDurationMapping.end()) {
                    idleDuration.store(it->second);
                }
            } else if (e.type == SDL_KEYUP) {
                keyIsPressed.store(false);
            }
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
    return (void*)0;
}


int main() {
    int stride = 128;
    int size = 1 << 18;
    float *d_data;

    cudaMalloc(&d_data, size * sizeof(float));
    cudaMemset(d_data, 0, size * sizeof(float));

    int blockSize = 256;
    int numBlocks = 2048*8; 

    // Create thread for non-blocking keyboard input
    pthread_t inputThread;
    if(pthread_create(&inputThread, nullptr, listenForKeyPress, nullptr)) {
        std::cerr << "Error creating input thread" << std::endl;
        return -1;
    }

    // Main loop for executing kernels
    while (true) {
        if (keyIsPressed.load()) {
            memoryIntensiveKernel<<<numBlocks, blockSize>>>(d_data, size, stride);
            cudaDeviceSynchronize();
            std::this_thread::sleep_for(std::chrono::microseconds(idleDuration.load()));
        }
    }

    // Cleanup
    cudaFree(d_data);
    cudaDeviceReset();
    pthread_join(inputThread, nullptr);

    return 0;
}
