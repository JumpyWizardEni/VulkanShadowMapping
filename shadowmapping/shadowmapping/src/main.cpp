#define GLFW_INCLUDE_VULKAN

#include "GLFW/glfw3.h"

#define TINYOBJLOADER_IMPLEMENTATION

#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stbi_load.h"
#include "cmath"

#ifdef WIN32
#pragma comment(lib, "glfw3.lib")
#endif

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/hash.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "vk_utils.h"

#define SHADOWMAP_DIM 1024
const int WIDTH = 800;
const int HEIGHT = 600;
float lastX = 400, lastY = 300;
float yaw = -90.0f;
float pitch = 0.0f;
bool firstMouse;
float deltaTime = 0.0f;
float lastFrame = 0.0f;
const int NUM = 4;
const int MODELS_COUNT = 4;
const int SCALES[4] = {5, 500, 20, 5};
bool keys[1024];
bool visualiseBuffer = false;
bool oldState = false;
const std::vector<std::string> MODELS = {"models/table.obj", "models/metal.obj", "models/spider.obj", "models/statue"
                                                                                                      ".obj"

};
const std::vector<std::string> TEXTURES = {"models/wood.png", "models/metal.jpg", "models/pink.jpg", "models/statue"
                                                                                                     ".jpg"};
const std::array<float, 4> offsetsX = {0, -0.5, -1, -0.3};
const std::array<float, 4> offsetsY = {0, 3, 1, 2};
const std::array<float, 4> offsetsZ = {0, 0, 0.5, 0.2};
const int MAX_FRAMES_IN_FLIGHT = 2;


const std::vector<const char *> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 lightMatrix;
    alignas(16) glm::vec3 lightPos;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }

    bool operator==(const Vertex &other) const {
        return pos == other.pos && normal == other.normal && texCoord == other.texCoord;
    }
};

namespace std {
    template<>
    struct hash<Vertex> {
        size_t operator()(Vertex const &vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}
glm::vec3 cameraPos = glm::vec3(1.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 lightPos = glm::vec3(1.5, 4.6, -0.86f);


#ifdef NDEBUGjpg
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


class HelloTriangleApplication {
public:
    void run() {
        InitWindow();

        InitVulkan();

        CreateResources();

        MainLoop();

        Cleanup();
    }

private:
    GLFWwindow *window;

    VkInstance instance;
    std::vector<const char *> enabledLayers;

    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    vk_utils::ScreenBufferResources screen;

    std::array<VkRenderPass, 2> renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    std::array<VkPipeline, 3> graphicsPipeline;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkImage> depthImage;
    VkImage shadowMapImage;
    std::vector<VkDeviceMemory> depthImageMemory;
    VkDeviceMemory shadowMapImageMemory;
    std::vector<VkImageView> depthImageView;
    VkImageView shadowMapImageView;
    std::vector<VkSampler> depthSampler;
    VkSampler shadowMapSampler;

    std::vector<VkImage> textureImage;
    std::vector<VkDeviceMemory> textureImageMemory;
    std::vector<VkImageView> textureImageView;
    std::vector<VkSampler> textureSampler;

    std::vector<std::vector<Vertex>> vertices;
    std::vector<VkBuffer> vertexBuffer;
    std::vector<VkDeviceMemory> vertexBufferMemory;

    std::vector<VkFramebuffer> depthFramebuffer;

    std::vector<uint32_t> indices;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    std::vector<std::vector<VkBuffer>> uniformBuffers;
    std::vector<std::vector<VkDeviceMemory>> uniformBuffersMemory;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    //Храним объемы памяти под каждый объект
    std::vector<size_t> sizesVert;
    std::vector<size_t> sizesInd;

    struct SyncObj {
        std::vector<VkSemaphore> imageAvailableSemaphores;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        std::vector<VkFence> inFlightFences;
        std::vector<VkFence> imagesInFlight;
    } m_sync;

    size_t currentFrame = 0;

    //По 1 и 2 меняем вид, остальные клавиши - движение камеры
    static void keyCallBack(GLFWwindow *window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_1) {
            visualiseBuffer = false;
            return;
        }
        if (key == GLFW_KEY_2) {
            visualiseBuffer = true;
            return;
        }
        if (action == GLFW_PRESS)
            keys[key] = true;
        else if (action == GLFW_RELEASE)
            keys[key] = false;
    }

    void doMovement() {
        GLfloat cameraSpeed = 1.05f * deltaTime;
        if (keys[GLFW_KEY_W])
            cameraPos += cameraSpeed * cameraFront;
        if (keys[GLFW_KEY_S])
            cameraPos -= cameraSpeed * cameraFront;
        if (keys[GLFW_KEY_A])
            cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        if (keys[GLFW_KEY_D])
            cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    }

    static void mouseCallBack(GLFWwindow *window, double xpos, double ypos) {
        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        GLfloat xoffset = xpos - lastX;
        GLfloat yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;

        GLfloat sensitivity = 0.05;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);
    }

    void InitWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetCursorPosCallback(window, mouseCallBack);
        glfwSetKeyCallback(window, keyCallBack);
    }


    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
            VkDebugReportFlagsEXT flags,
            VkDebugReportObjectTypeEXT objectType,
            uint64_t object,
            size_t location,
            int32_t messageCode,
            const char *pLayerPrefix,
            const char *pMessage,
            void *pUserData) {
        printf("[Debug Report]: %s: %s\n", pLayerPrefix, pMessage);
        return VK_FALSE;
    }

    VkDebugReportCallbackEXT debugReportCallback;


    void InitVulkan() {
        const int deviceId = 0;

        std::vector<const char *> extensions;
        {
            uint32_t glfwExtensionCount = 0;
            const char **glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
            extensions = std::vector<const char *>(glfwExtensions,
                                                   glfwExtensions + glfwExtensionCount);
        }

        instance = vk_utils::CreateInstance(enableValidationLayers, enabledLayers, extensions);
        if (enableValidationLayers)
            vk_utils::InitDebugReportCallback(instance, &debugReportCallbackFn,
                                              &debugReportCallback);

        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("glfwCreateWindowSurface: failed to create window surface!");

        physicalDevice = vk_utils::FindPhysicalDevice(instance, true, deviceId);
        auto queueFID = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT);

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFID, surface, &presentSupport);
        if (!presentSupport)
            throw std::runtime_error(
                    "vkGetPhysicalDeviceSurfaceSupportKHR: no present support for the target device and graphics queue");

        device = vk_utils::CreateLogicalDevice(queueFID, physicalDevice, enabledLayers,
                                               deviceExtensions);
        vkGetDeviceQueue(device, queueFID, 0, &graphicsQueue);
        vkGetDeviceQueue(device, queueFID, 0, &presentQueue);

        {
            VkCommandPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.queueFamilyIndex = vk_utils::GetQueueFamilyIndex(physicalDevice,
                                                                      VK_QUEUE_GRAPHICS_BIT);

            if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
                throw std::runtime_error(
                        "[CreateCommandPoolAndBuffers]: failed to create command pool!");
        }

        vk_utils::CreateSwapChain(physicalDevice, device, surface, WIDTH, HEIGHT,
                                  &screen);

        vk_utils::CreateScreenImageViews(device, &screen);
    }

    void CreateResources() {
        depthImage.resize(MODELS_COUNT);
        depthImageMemory.resize(MODELS_COUNT);
        depthImageView.resize(MODELS_COUNT);
        depthSampler.resize(MODELS_COUNT);

        textureImage.resize(MODELS_COUNT);
        textureImageMemory.resize(MODELS_COUNT);
        textureImageView.resize(MODELS_COUNT);
        textureSampler.resize(MODELS_COUNT);


        vertices.resize(MODELS_COUNT);
        vertexBuffer.resize(MODELS_COUNT);
        vertexBufferMemory.resize(MODELS_COUNT);


        CreateShadowRenderPass(physicalDevice, device, screen.swapChainImageFormat,
                               &renderPass[0]);
        CreateRenderPass(physicalDevice, device, screen.swapChainImageFormat,
                         &renderPass[1]);

        createDescriptorSetLayout(device, &descriptorSetLayout);


        CreateGraphicsPipeline(device, screen.swapChainExtent, renderPass,
                               &pipelineLayout, graphicsPipeline, descriptorSetLayout);
        createDepthResources(physicalDevice, device, shadowMapImage, shadowMapImageView, shadowMapSampler,
                             shadowMapImageMemory,
                             screen, SHADOWMAP_DIM, SHADOWMAP_DIM);
        for (size_t i = 0; i < MODELS_COUNT; ++i) {
            createDepthResources(physicalDevice, device, depthImage[i], depthImageView[i], depthSampler[i],
                                 depthImageMemory[i],
                                 screen, screen.swapChainExtent.width, screen.swapChainExtent.height);

        }
        CreateScreenFrameBuffersForDepth(device, depthFramebuffer, renderPass[0], &screen, shadowMapImageView,
                                         SHADOWMAP_DIM);
        CreateScreenFrameBuffers(device, renderPass[1], &screen, depthImageView);
        size_t offsetVert = 0;
        size_t offsetInd = 0;
        for (size_t i = 0; i < MODELS_COUNT; ++i) {
            createTextureImage(physicalDevice, device, textureImage[i], textureImageMemory[i], graphicsQueue,
                               commandPool, i);
            createTextureImageView(device, textureImageView[i], textureImage[i]);
            createTextureSampler(&textureSampler[i]);
            loadModel(vertices[i], indices, i, sizesVert, sizesInd, offsetVert, offsetInd);
            createVertexBuffer(physicalDevice, device, graphicsQueue, commandPool, vertices[i], vertexBuffer[i],
                               vertexBufferMemory[i]);

        }
        createIndexBuffer(physicalDevice, device, graphicsQueue, commandPool, indexBuffer, indexBufferMemory,
                          indices, offsetInd);

        createUniformBuffers(physicalDevice, device, uniformBuffers, uniformBuffersMemory, screen);

        createDescriptorPool(device, &descriptorPool, screen);
        createDescriptorSets(device, descriptorPool, screen, descriptorSetLayout, descriptorSets, uniformBuffers,
                             textureImageView, textureSampler, shadowMapImageView,
                             shadowMapSampler);
        createCommandBuffers(device, commandBuffers, commandPool, screen, indices, renderPass,
                             graphicsPipeline,
                             vertexBuffer, indexBuffer, depthFramebuffer, pipelineLayout, descriptorSets, sizesVert,
                             sizesInd);


        CreateSyncObjects(device, &m_sync, screen);
    }


    void MainLoop() {
        float currentFrame;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            currentFrame = glfwGetTime();
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;
            doMovement();
            DrawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void Cleanup() {


        if (enableValidationLayers) {
            // destroy callback.
            auto func = (PFN_vkDestroyDebugReportCallbackEXT) vkGetInstanceProcAddr(instance,
                                                                                    "vkDestroyDebugReportCallbackEXT");
            if (func == nullptr)
                throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
            func(instance, debugReportCallback, NULL);
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, m_sync.renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, m_sync.imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, m_sync.inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);



        for (size_t i = 0; i < graphicsPipeline.size(); ++i) {
            vkDestroyPipeline(device, graphicsPipeline[i], nullptr);

        }
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        for (size_t i = 0; i < renderPass.size(); ++i) {
            vkDestroyRenderPass(device, renderPass[i], nullptr);
        }
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);


        for (size_t i = 0; i < MODELS_COUNT; ++i) {
            vkFreeMemory(device, vertexBufferMemory[i], nullptr);
            vkDestroyBuffer(device, vertexBuffer[i], nullptr);
        }
        for (size_t i = 0; i < screen.swapChainImages.size(); i++) {

            for (int j = 0; j < MODELS_COUNT; ++j) {
                vkFreeMemory(device, uniformBuffersMemory[j][i], nullptr);
                vkDestroyBuffer(device, uniformBuffers[j][i], nullptr);
            }
        }
        vkFreeMemory(device, shadowMapImageMemory, nullptr);
        vkDestroySampler(device, shadowMapSampler, nullptr);
        vkDestroyImageView(device, shadowMapImageView, nullptr);
        vkDestroyImage(device, shadowMapImage, nullptr);

        for (size_t i = 0; i < MODELS_COUNT; ++i) {
            vkFreeMemory(device, textureImageMemory[i], nullptr);
            vkFreeMemory(device, depthImageMemory[i], nullptr);
            vkDestroySampler(device, textureSampler[i], nullptr);
            vkDestroySampler(device, depthSampler[i], nullptr);
            vkDestroyImageView(device, textureImageView[i], nullptr);
            vkDestroyImageView(device, depthImageView[i], nullptr);
            vkDestroyImage(device, textureImage[i], nullptr);
            vkDestroyImage(device, depthImage[i], nullptr);

        }


        vkFreeMemory(device, indexBufferMemory, nullptr);
        vkDestroyBuffer(device, indexBuffer, nullptr);

        for (auto framebuffer : screen.swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (size_t i = 0; i < screen.swapChainImageViews.size(); ++i) {
            vkDestroyFramebuffer(device, depthFramebuffer[i], nullptr);
        }

        for (auto imageView : screen.swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, screen.swapChain, nullptr);
        vkDestroyDevice(device, nullptr);

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    static VkFormat findDepthFormat(VkPhysicalDevice physDevice) {
        return findSupportedFormat(
                {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                VK_IMAGE_TILING_OPTIMAL,
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
                physDevice);
    }

    static VkFormat
    findSupportedFormat(const std::vector<VkFormat> &candidates, VkImageTiling tiling,
                        VkFormatFeatureFlags features, VkPhysicalDevice physDevice) {
        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR &&
                (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                       (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    static void createTextureImage(VkPhysicalDevice physDevice, VkDevice device, VkImage &textureImage, VkDeviceMemory
    &textureImageMemory, VkQueue &graphicsQueue,
                                   VkCommandPool &commandPool, size_t ind) {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load(TEXTURES[ind % NUM].c_str(), &texWidth, &texHeight, &texChannels,
                                    STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(physDevice, device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void *data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        stbi_image_free(pixels);

        createImage(VK_IMAGE_LAYOUT_UNDEFINED, physDevice, device, texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, graphicsQueue, device, commandPool);
        copyBufferToImage(device, graphicsQueue, commandPool, stagingBuffer, textureImage, static_cast<uint32_t>
                          (texWidth),
                          static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, graphicsQueue, device, commandPool);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    static void createTextureImageView(VkDevice &device, VkImageView &textureImageView, VkImage &textureImage) {
        textureImageView = createImageView(device, textureImage, VK_FORMAT_R8G8B8A8_SRGB,
                                           VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void createTextureSampler(VkSampler *textureSampler) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        if (vkCreateSampler(device, &samplerInfo, nullptr, textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    static VkImageView createImageView(VkDevice &device, VkImage &image, VkFormat format,
                                       VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }

    static void createImage(VkImageLayout layout, VkPhysicalDevice physDevice, VkDevice device, uint32_t width,
                            uint32_t height,
                            VkFormat
                            format,
                            VkImageTiling tiling, VkImageUsageFlags usage,
                            VkMemoryPropertyFlags properties, VkImage &image,
                            VkDeviceMemory &imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = layout;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(physDevice, memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    static void transitionImageLayout(VkImage &image, VkFormat format, VkImageLayout oldLayout,
                                      VkImageLayout newLayout, VkQueue &graphicsQueue, VkDevice &device,
                                      VkCommandPool &commandPool) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool, device);

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
                commandBuffer,
                sourceStage, destinationStage,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

        endSingleTimeCommands(commandBuffer, graphicsQueue, device, commandPool);
    }

    static void createDescriptorPool(VkDevice &device, VkDescriptorPool *descriptorPool,
                                     vk_utils::ScreenBufferResources &screen) {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(screen.swapChainImages.size() * MODELS_COUNT);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(screen.swapChainImages.size() * MODELS_COUNT * 2);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(screen.swapChainImages.size() * MODELS_COUNT);

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    static void createDescriptorSets(VkDevice &device, VkDescriptorPool &descriptorPool,
                                     vk_utils::ScreenBufferResources &screen, VkDescriptorSetLayout
                                     &descriptorSetLayout, std::vector<VkDescriptorSet> &descriptorSets,
                                     std::vector<std::vector<VkBuffer>> &uniformBuffers, std::vector<VkImageView>
                                     &textureImageView,
                                     std::vector<VkSampler>
                                     &textureSampler,
                                     VkImageView
                                     &depthImageView,
                                    VkSampler
                                     &depthSampler) {
        std::vector<VkDescriptorSetLayout> layouts(screen.swapChainImages.size() * MODELS_COUNT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(screen.swapChainImages.size() * MODELS_COUNT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(screen.swapChainImages.size() * MODELS_COUNT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }


        for (size_t i = 0; i < screen.swapChainImages.size(); i++) {

            for (int j = 0; j < MODELS_COUNT; ++j) {
                std::vector<VkWriteDescriptorSet> descriptorWrites;
                descriptorWrites.resize(3);
                VkDescriptorBufferInfo bufferInfo{};
                bufferInfo.buffer = uniformBuffers[j][i];
                bufferInfo.offset = 0;
                bufferInfo.range = sizeof(UniformBufferObject);

                VkDescriptorImageInfo imageInfo{};
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.imageView = textureImageView[j];
                imageInfo.sampler = textureSampler[j];


                VkDescriptorImageInfo depthInfo{};
                depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                depthInfo.imageView = depthImageView;
                depthInfo.sampler = depthSampler;


                descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[0].dstSet = descriptorSets[j + (i * (MODELS_COUNT))];
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &bufferInfo;

                descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[1].dstSet = descriptorSets[j + (i * (MODELS_COUNT))];
                descriptorWrites[1].dstBinding = 1;
                descriptorWrites[1].dstArrayElement = 0;
                descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrites[1].descriptorCount = 1;
                descriptorWrites[1].pImageInfo = &imageInfo;

                descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[2].dstSet = descriptorSets[j + (i * (MODELS_COUNT))];
                descriptorWrites[2].dstBinding = 2;
                descriptorWrites[2].dstArrayElement = 0;
                descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrites[2].descriptorCount = 1;
                descriptorWrites[2].pImageInfo = &depthInfo;


                vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(),
                                       0,
                                       nullptr);

            }


        }

    }

    static void createDepthResources(VkPhysicalDevice physDevice, VkDevice device,
                                     VkImage &depthImage, VkImageView &depthImageView, VkSampler
                                     &depthSampler, VkDeviceMemory &depthImageMemory,
                                     vk_utils::ScreenBufferResources
                                     &screen, size_t width, size_t height) {
        VkFormat depthFormat = VK_FORMAT_D16_UNORM;

        createImage(VK_IMAGE_LAYOUT_UNDEFINED, physDevice, device, width, height, depthFormat,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(device, depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_NEAREST;
        sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = sampler.addressModeU;
        sampler.addressModeW = sampler.addressModeU;
        sampler.mipLodBias = 0.0f;
        sampler.maxAnisotropy = 1.0f;
        sampler.minLod = 0.0f;
        sampler.maxLod = 1.0f;
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        vkCreateSampler(device, &sampler, nullptr, &depthSampler);
    }

    static void copyBufferToImage(VkDevice &device, VkQueue &graphicsQueue, VkCommandPool &commandPool, VkBuffer
    &buffer,
                                  VkImage &image, uint32_t width, uint32_t
                                  height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool, device);

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {
                width,
                height,
                1};

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &region);

        endSingleTimeCommands(commandBuffer, graphicsQueue, device, commandPool);
    }

    static void loadModel(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, size_t ind,
                          std::vector<size_t> &sizesVert, std::vector<size_t> &sizesInd, size_t &offsetVert, size_t
                          &offsetInd) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODELS[ind % NUM].c_str())) {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto &shape : shapes) {
            for (const auto &index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {
                        offsetsX[ind] + attrib.vertices[3 * index
                                .vertex_index +
                                                        0] /
                                        SCALES[ind],
                        offsetsY[ind] + attrib.vertices[3 * index.vertex_index + 1] / SCALES[ind],
                        offsetsZ[ind] + attrib.vertices[3 * index.vertex_index + 2] / SCALES[ind]};
                if (index.texcoord_index >= 0) {
                    vertex.texCoord = {
                            attrib.texcoords[2 * index.texcoord_index + 0],
                            1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};
                }

                if (index.normal_index >= 0) {
                    vertex.normal = {attrib.normals[3 * index.normal_index + 0] / SCALES[ind], attrib.normals[3 * index
                            .normal_index + 1] / SCALES[ind],
                                     attrib.normals[3 * index.normal_index + 2] / SCALES[ind]};
                }

                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }
        sizesVert.push_back(vertices.size());
        sizesInd.push_back(indices.size() - offsetInd);
        offsetInd = indices.size();
        offsetVert = vertices.size();
    }

    static void createVertexBuffer(VkPhysicalDevice physDevice, VkDevice device, VkQueue
    &graphicsQueue, VkCommandPool &commandPool, std::vector<Vertex> &vertices,
                                   VkBuffer &vertexBuffer, VkDeviceMemory &vertexBufferMemory) {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(physDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void *data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(physDevice, device, bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize, device, graphicsQueue, commandPool);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    static void createIndexBuffer(VkPhysicalDevice physDevice, VkDevice device, VkQueue
    &graphicsQueue, VkCommandPool &commandPool, VkBuffer &indexBuffer, VkDeviceMemory
                                  &indexBufferMemory, std::vector<uint32_t> &indices, size_t neededSize) {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(physDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void *data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(physDevice, device, bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize, device, graphicsQueue, commandPool);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    static void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDevice device, VkQueue
    &graphicsQueue, VkCommandPool &commandPool) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool, device);

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer, graphicsQueue, device, commandPool);
    }


    static void createUniformBuffers(VkPhysicalDevice physDevice, VkDevice device, std::vector<std::vector<VkBuffer>>
    &uniformBuffers,
                                     std::vector<std::vector<VkDeviceMemory>> &uniformBuffersMemory,
                                     vk_utils::ScreenBufferResources &screen) {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);
        size_t size = screen.swapChainImages.size() * MODELS_COUNT;
        uniformBuffers.resize(MODELS_COUNT);
        uniformBuffersMemory.resize(MODELS_COUNT);

        for (size_t i = 0; i < MODELS_COUNT; ++i) {
            uniformBuffers[i].resize(screen.swapChainImages.size());
            uniformBuffersMemory[i].resize(screen.swapChainImages.size());
            for (size_t j = 0; j < screen.swapChainImages.size(); ++j) {
                createBuffer(physDevice, device, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             uniformBuffers[i][j], uniformBuffersMemory[i][j]);
            }
        }
    }

    static void createBuffer(VkPhysicalDevice physDevice, VkDevice device, VkDeviceSize size,
                             VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                             VkBuffer &buffer, VkDeviceMemory &bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(physDevice, memRequirements.memoryTypeBits,
                                                   properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                                   VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    static void CreateRenderPass(VkPhysicalDevice physDevice, VkDevice a_device,
                                 VkFormat a_swapChainImageFormat,
                                 VkRenderPass *a_pRenderPass) {
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = a_swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment = {};
        depthAttachment.format = VK_FORMAT_D16_UNORM;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


        VkAttachmentReference depthAttachmentRef = {};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


        VkSubpassDescription subpassColor = {};
        subpassColor.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassColor.colorAttachmentCount = 1;
        subpassColor.pColorAttachments = &colorAttachmentRef;
        subpassColor.pDepthStencilAttachment = &depthAttachmentRef;

        std::array<VkSubpassDependency, 2> dependencies;
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;


        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassColor;
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();

        if (vkCreateRenderPass(a_device, &renderPassInfo, nullptr, a_pRenderPass) != VK_SUCCESS)
            throw std::runtime_error("[CreateRenderPass]: failed to create render pass!");
    }

    static void CreateShadowRenderPass(VkPhysicalDevice physDevice, VkDevice a_device,
                                       VkFormat a_swapChainImageFormat,
                                       VkRenderPass *a_pRenderPass) {
        VkAttachmentDescription attachmentDescription{};
        attachmentDescription.format = VK_FORMAT_D16_UNORM;
        attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference depthReference = {};
        depthReference.attachment = 0;
        depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 0;
        subpass.pDepthStencilAttachment = &depthReference;

        std::array<VkSubpassDependency, 2> dependencies{};

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo renderPassCreateInfo{};
        renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassCreateInfo.attachmentCount = 1;
        renderPassCreateInfo.pAttachments = &attachmentDescription;
        renderPassCreateInfo.subpassCount = 1;
        renderPassCreateInfo.pSubpasses = &subpass;
        renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassCreateInfo.pDependencies = dependencies.data();

        if (vkCreateRenderPass(a_device, &renderPassCreateInfo, nullptr, a_pRenderPass) != VK_SUCCESS)
            throw std::runtime_error("[CreateRenderPass]: failed to create render pass!");
    }

    static void
    createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout *descriptorSetLayout) {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding depthLayoutBinding{};
        depthLayoutBinding.binding = 2;
        depthLayoutBinding.descriptorCount = 1;
        depthLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        depthLayoutBinding.pImmutableSamplers = nullptr;
        depthLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {uboLayoutBinding,
                                                                samplerLayoutBinding, depthLayoutBinding};
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    static void
    CreateGraphicsPipeline(VkDevice a_device, VkExtent2D a_screenExtent, std::array<VkRenderPass, 2> &a_renderPass,
                           VkPipelineLayout *a_pLayout, std::array<VkPipeline, 3> &a_pPipiline, VkDescriptorSetLayout
                           &descriptorSetLayout) {
        auto vertShaderCode = vk_utils::ReadFile("shaders/vert.spv");
        auto fragShaderCode = vk_utils::ReadFile("shaders/frag.spv");
        auto vertDepthCode = vk_utils::ReadFile("shaders/vertDepth.spv");
        auto fragDepthCode = vk_utils::ReadFile("shaders/fragDepth.spv");
        auto bufferVertCode = vk_utils::ReadFile("shaders/bufferVert.spv");
        auto bufferFragCode = vk_utils::ReadFile("shaders/bufferFrag.spv");

        VkShaderModule vertShaderModule = vk_utils::CreateShaderModule(a_device, vertShaderCode);
        VkShaderModule fragShaderModule = vk_utils::CreateShaderModule(a_device, fragShaderCode);
        VkShaderModule vertDepthModule = vk_utils::CreateShaderModule(a_device, vertDepthCode);
        VkShaderModule fragDepthModule = vk_utils::CreateShaderModule(a_device, fragDepthCode);
        VkShaderModule bufferVertModule = vk_utils::CreateShaderModule(a_device, bufferVertCode);
        VkShaderModule bufferFragModule = vk_utils::CreateShaderModule(a_device, bufferFragCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";


        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) a_screenExtent.width;
        viewport.height = (float) a_screenExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = a_screenExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        std::vector<VkDynamicState> dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.flags = 0;
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        if (vkCreatePipelineLayout(a_device, &pipelineLayoutInfo, nullptr, a_pLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = *a_pLayout;
        pipelineInfo.renderPass = a_renderPass[1];
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(a_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &a_pPipiline[1]) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
        vertShaderStageInfo.module = bufferVertModule;
        fragShaderStageInfo.module = bufferFragModule;
        shaderStages[0] = vertShaderStageInfo;
        shaderStages[1] = fragShaderStageInfo;
        pipelineInfo.pStages = shaderStages;

        if (vkCreateGraphicsPipelines(a_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &a_pPipiline[2]) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vertShaderStageInfo.module = vertDepthModule;
        pipelineInfo.stageCount = 1;
        pipelineInfo.pStages = &vertShaderStageInfo;
        colorBlending.attachmentCount = 0;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        pipelineInfo.renderPass = a_renderPass[0];


        rasterizer.depthBiasEnable = VK_TRUE;
        dynamicStateEnables.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
        dynamicState.dynamicStateCount = dynamicStateEnables.size();
        dynamicState.pDynamicStates = dynamicStateEnables.data();
        if (vkCreateGraphicsPipelines(a_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &a_pPipiline[0]) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }


        vkDestroyShaderModule(a_device, fragShaderModule, nullptr);
        vkDestroyShaderModule(a_device, vertShaderModule, nullptr);
        vkDestroyShaderModule(a_device, fragDepthModule, nullptr);
        vkDestroyShaderModule(a_device, vertDepthModule, nullptr);
        vkDestroyShaderModule(a_device, bufferFragModule, nullptr);
        vkDestroyShaderModule(a_device, bufferVertModule, nullptr);
    }

    static VkCommandBuffer beginSingleTimeCommands(VkCommandPool &commandPool, VkDevice device) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    static void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkQueue &graphicsQueue, VkDevice &device,
                                      VkCommandPool &commandPool) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    static void CreateSyncObjects(VkDevice a_device, SyncObj *a_pSyncObjs, vk_utils::ScreenBufferResources &screen) {
        a_pSyncObjs->imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        a_pSyncObjs->renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        a_pSyncObjs->imagesInFlight.resize(screen.swapChainImages.size() * MODELS_COUNT, VK_NULL_HANDLE);
        a_pSyncObjs->inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(a_device, &semaphoreInfo, nullptr,
                                  &a_pSyncObjs->imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(a_device, &semaphoreInfo, nullptr,
                                  &a_pSyncObjs->renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(a_device, &fenceInfo, nullptr, &a_pSyncObjs->inFlightFences[i]) !=
                VK_SUCCESS) {
                throw std::runtime_error(
                        "[CreateSyncObjects]: failed to create synchronization objects for a frame!");
            }
        }
    }

    static void
    updateUniformBuffer(VkDevice device, std::vector<VkDeviceMemory> &uniformBuffersMemory,
                        uint32_t currentImage, vk_utils::ScreenBufferResources &screen, size_t objectNumber) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(
                currentTime - startTime).count();

        UniformBufferObject ubo{};

        if (objectNumber == 1) {
            ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
                                    glm::vec3(0.0f, 1.0f, 0.0f));
        } else {
            ubo.model = glm::mat4(1.0);

        }

        ubo.view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        ubo.proj = glm::perspective(glm::radians(60.0f), screen.swapChainExtent.width /
                                                         (float) screen.swapChainExtent.height,
                                    0.1f, 256.0f);
        ubo.proj[1][1] *= -1;
        float nearPlane = 0.1f, farPlane = 256.0f;
        ubo.lightPos = lightPos;
        glm::mat4 depthProjectionMatrix = glm::perspective(glm::radians(60.0f), screen.swapChainExtent.width /
                                                                                (float) screen.swapChainExtent.height,
                                                           nearPlane, farPlane);
        glm::mat4 depthViewMatrix = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0, 1, 0));
        glm::mat4 depthModelMatrix = ubo.model;

        depthProjectionMatrix[1][1] *= -1;
        ubo.lightMatrix = depthProjectionMatrix * depthViewMatrix * depthModelMatrix;


        void *data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
    }

    static void createCommandBuffers(VkDevice &device, std::vector<VkCommandBuffer> &commandBuffers, VkCommandPool
    &commandPool,
                                     vk_utils::ScreenBufferResources &screen, std::vector<uint32_t>
                                     &indices,
                                     std::array<VkRenderPass, 2> &renderPass, std::array<VkPipeline, 3>
                                     &graphicsPipeline, std::vector<VkBuffer>
                                     &vertexBuffer,
                                     VkBuffer &indexBuffer, std::vector<VkFramebuffer> &depthFramebuffer,
                                     VkPipelineLayout &pipelineLayout,
                                     std::vector<VkDescriptorSet> &descriptorSets, std::vector<size_t> &sizesVert,
                                     std::vector<size_t> &sizesInd) {
        commandBuffers.resize(screen.swapChainFramebuffers.size());

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();
        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
        size_t currIndex = 0;
        VkViewport viewport;
        viewport.minDepth = 0.0f;
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.maxDepth = 1.0f;
        VkRect2D scissor;
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        for (size_t i = 0; i < commandBuffers.size(); i++) {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }
            {
                VkRenderPassBeginInfo renderPassInfo{};
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.renderPass = renderPass[0];
                renderPassInfo.framebuffer = depthFramebuffer[0];
                renderPassInfo.renderArea.offset = {0, 0};

                renderPassInfo.renderArea.extent.width = SHADOWMAP_DIM;
                renderPassInfo.renderArea.extent.height = SHADOWMAP_DIM;

                viewport.width = (float) SHADOWMAP_DIM;
                viewport.height = (float) SHADOWMAP_DIM;

                vkCmdSetViewport(commandBuffers[i], 0, 1, &viewport);

                scissor.extent.width = SHADOWMAP_DIM;
                scissor.extent.height = SHADOWMAP_DIM;

                vkCmdSetScissor(commandBuffers[i], 0, 1, &scissor);
                vkCmdSetDepthBias(
                        commandBuffers[i],
                        1.25,
                        0.0f,
                        1.75);
                std::array<VkClearValue, 1> clearValues{};
                clearValues[0].depthStencil = {1.0f, 0};

                renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
                renderPassInfo.pClearValues = clearValues.data();

                vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline[0]);
                vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);
                size_t offset = 0;


                for (size_t j = 0; j < MODELS_COUNT; ++j) {
                    VkBuffer vertexBuffers[] = {vertexBuffer[j]};
                    VkDeviceSize offsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
                    vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                            &descriptorSets[MODELS_COUNT * i + j], 0, nullptr);

                    vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(sizesInd[j]), 1, offset, 0, 0);
                    offset += sizesInd[j];

                }
                vkCmdEndRenderPass(commandBuffers[i]);
            }

            {
                VkRenderPassBeginInfo renderPassInfo{};
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.renderPass = renderPass[1];
                renderPassInfo.framebuffer = screen.swapChainFramebuffers[i];
                renderPassInfo.renderArea.offset = {0, 0};
                renderPassInfo.renderArea.extent = screen.swapChainExtent;

                std::array<VkClearValue, 2> clearValues{};

                clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
                clearValues[1].depthStencil = {1.0f, 0};

                renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
                renderPassInfo.pClearValues = clearValues.data();
                viewport.width = (float) screen.swapChainExtent.width;
                viewport.height = (float) screen.swapChainExtent.height;

                vkCmdSetViewport(commandBuffers[i], 0, 1, &viewport);
                scissor.extent.width = screen.swapChainExtent.width;
                scissor.extent.height = screen.swapChainExtent.height;

                vkCmdSetScissor(commandBuffers[i], 0, 1, &scissor);
                vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                size_t offset = 0;
                size_t graphPipe = visualiseBuffer ? 2 : 1;
                vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline[graphPipe]);
                vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);
                for (size_t j = 0; j < MODELS_COUNT; ++j) {
                    VkBuffer vertexBuffers[] = {vertexBuffer[j]};
                    VkDeviceSize offsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
                    vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                            &descriptorSets[MODELS_COUNT * i + j], 0, nullptr);

                    vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(sizesInd[j]), 1, offset, 0, 0);
                    offset += sizesInd[j];

                }
                vkCmdEndRenderPass(commandBuffers[i]);
            }

            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }

    }

    void DrawFrame() {
        if (visualiseBuffer != oldState) {
            oldState = visualiseBuffer;
            recreateCommandBuffer();
        }
        vkWaitForFences(device, 1, &m_sync.inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, screen.swapChain, UINT64_MAX, m_sync
                                      .imageAvailableSemaphores[currentFrame],
                              VK_NULL_HANDLE, &imageIndex);

        for (size_t i = 0; i < MODELS_COUNT; ++i) {
            updateUniformBuffer(device, uniformBuffersMemory[i], imageIndex, screen, i);
        }

        if (m_sync.imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &m_sync.imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }
        m_sync.imagesInFlight[imageIndex] = m_sync.inFlightFences[currentFrame];

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {m_sync.imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        VkSemaphore signalSemaphores[] = {m_sync.renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences(device, 1, &m_sync.inFlightFences[currentFrame]);

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, m_sync.inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {screen.swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;

        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(presentQueue, &presentInfo);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void recreateCommandBuffer() {
        createCommandBuffers(device, commandBuffers, commandPool, screen, indices, renderPass,
                             graphicsPipeline,
                             vertexBuffer, indexBuffer, depthFramebuffer, pipelineLayout, descriptorSets, sizesVert,
                             sizesInd);
    }

};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}