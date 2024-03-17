#pragma once

#include "fmt/base.h"
#include <GLFW/glfw3.h>
#include <vk_types.h>

constexpr float CAMERA_SPEED = 0.3f;

class Camera {
public:
  Camera() = default;

  Camera(GLFWwindow* window) {
    fmt::println("constructing camera");
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_callback);
  }

  inline static glm::vec3 velocity;
  inline static glm::vec3 position;
  inline static float pitch;
  inline static float yaw;
  inline static double cursor_x, cursor_y;

  glm::mat4 get_view_matrix();
  glm::mat4 get_rotation_matrix();
  void update();
  void process_glfw_key(int key, int scancode, int action, int mods);
  void process_glfw_cursor(double xpos, double ypos);

private:
  static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

    Camera* obj = (Camera*)glfwGetWindowUserPointer(window);
    obj->process_glfw_key(key, scancode, action, mods);
  }
  static void cursor_callback(GLFWwindow* window, double xpos, double ypos) {
    Camera* obj = (Camera*)glfwGetWindowUserPointer(window);
    obj->process_glfw_cursor(xpos, ypos);
  };
};
