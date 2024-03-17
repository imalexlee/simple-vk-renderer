#define GLM_ENABLE_EXPERIMENTAL

#include <GLFW/glfw3.h>
#include <camera.h>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

void Camera::update() {
  glm::mat4 camera_rotation = get_rotation_matrix();
  position += glm::vec3(camera_rotation * glm::vec4(velocity * 0.5f, 0.f));
};

void Camera::process_glfw_key(int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS) {
    if (key == GLFW_KEY_W) {
      velocity.z = -CAMERA_SPEED;
    }
    if (key == GLFW_KEY_A) {
      velocity.x = -CAMERA_SPEED;
    }
    if (key == GLFW_KEY_S) {
      velocity.z = CAMERA_SPEED;
    }
    if (key == GLFW_KEY_D) {
      velocity.x = CAMERA_SPEED;
    }
  }
  if (action == GLFW_RELEASE) {
    if (key == GLFW_KEY_W) {
      velocity.z = 0;
    }
    if (key == GLFW_KEY_A) {
      velocity.x = 0;
    }
    if (key == GLFW_KEY_S) {
      velocity.z = 0;
    }
    if (key == GLFW_KEY_D) {
      velocity.x = 0;
    }
  }
}

void Camera::process_glfw_cursor(double xpos, double ypos) {

  double rel_x = cursor_x - xpos;
  double rel_y = cursor_y - ypos;
  cursor_x = xpos;
  cursor_y = ypos;
  yaw -= (float)rel_x / 1000.f;
  pitch += (float)rel_y / 1000.f;
}

glm::mat4 Camera::get_view_matrix() {
  // move world in opposite direction to the camera
  glm::mat4 camera_translation = glm::translate(glm::mat4{1.f}, position);
  glm::mat4 camera_rotation = get_rotation_matrix();
  return glm::inverse(camera_translation * camera_rotation);
}

glm::mat4 Camera::get_rotation_matrix() {
  glm::quat pitch_rotation = glm::angleAxis(pitch, glm::vec3{1, 0, 0});
  glm::quat yaw_rotation = glm::angleAxis(yaw, glm::vec3{0, -1, 0});

  return glm::toMat4(yaw_rotation) * glm::toMat4(pitch_rotation);
}
