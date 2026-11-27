/* SPDX-FileCopyrightText: 2026 EX-CAD Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace blender {
struct bContext;
}

namespace blender::ed::excad {

class AiBridgeServer {
 public:
  static AiBridgeServer &instance();

  bool start(int port = 9877);
  void stop();
  bool is_running() const
  {
    return running_.load();
  }
  int port() const
  {
    return port_;
  }

  /* Must be called on Blender's main thread (e.g. via modal timer or poll operator). */
  int process_pending(bContext *C);

 private:
  AiBridgeServer() = default;
  ~AiBridgeServer();

  void server_loop();

  struct PendingRequest {
    std::string payload;
    std::function<void(const std::string &)> reply_fn;
  };

  std::atomic<bool> running_{false};
  int port_ = 9877;
  std::thread accept_thread_;
  std::mutex queue_mutex_;
  std::queue<PendingRequest> queue_;
};

}  // namespace blender::ed::excad
