/* SPDX-FileCopyrightText: 2026 EX-CAD Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
using excad_socket_t = SOCKET;
constexpr excad_socket_t kInvalidSocket = INVALID_SOCKET;
inline void close_socket(excad_socket_t s)
{
  closesocket(s);
}
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/select.h>
#  include <sys/socket.h>
#  include <unistd.h>
using excad_socket_t = int;
constexpr excad_socket_t kInvalidSocket = -1;
inline void close_socket(excad_socket_t s)
{
  close(s);
}
#endif

#include "excad_ai_bridge.hh"
#include "excad_intern.hh"
#include "excad_sdf_mesh.hh"
#include "excad_sketch_solver.hh"

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <future>
#include <sstream>
#include <string>

namespace blender::ed::excad {

namespace {

std::string json_extract_string(const std::string &json,
                                const std::string &key,
                                const std::string &def = "")
{
  const std::string pattern = "\"" + key + "\"";
  size_t pos = json.find(pattern);
  if (pos == std::string::npos) {
    return def;
  }
  pos = json.find(':', pos + pattern.size());
  if (pos == std::string::npos) {
    return def;
  }
  ++pos;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  if (pos >= json.size() || json[pos] != '"') {
    return def;
  }
  ++pos;
  std::string out;
  while (pos < json.size() && json[pos] != '"') {
    if (json[pos] == '\\' && pos + 1 < json.size()) {
      ++pos;
      if (json[pos] == 'n') {
        out.push_back('\n');
      }
      else {
        out.push_back(json[pos]);
      }
    }
    else {
      out.push_back(json[pos]);
    }
    ++pos;
  }
  return out;
}

double json_extract_double(const std::string &json, const std::string &key, double def = 0.0)
{
  const std::string pattern = "\"" + key + "\"";
  size_t pos = json.find(pattern);
  if (pos == std::string::npos) {
    return def;
  }
  pos = json.find(':', pos + pattern.size());
  if (pos == std::string::npos) {
    return def;
  }
  ++pos;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  if (pos >= json.size()) {
    return def;
  }
  char *endptr = nullptr;
  const double val = std::strtod(json.c_str() + pos, &endptr);
  if (endptr == json.c_str() + pos) {
    return def;
  }
  return val;
}

int json_extract_int(const std::string &json, const std::string &key, int def = 0)
{
  return static_cast<int>(json_extract_double(json, key, static_cast<double>(def)));
}

bool json_extract_bool(const std::string &json, const std::string &key, bool def = false)
{
  const std::string pattern = "\"" + key + "\"";
  size_t pos = json.find(pattern);
  if (pos == std::string::npos) {
    return def;
  }
  pos = json.find(':', pos + pattern.size());
  if (pos == std::string::npos) {
    return def;
  }
  ++pos;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  if (json.compare(pos, 4, "true") == 0) {
    return true;
  }
  if (json.compare(pos, 5, "false") == 0) {
    return false;
  }
  return def;
}

std::string dispatch_command(bContext *C, const std::string &payload)
{
  const std::string cmd = json_extract_string(payload, "cmd");
  SketchDocument &doc = SketchDocument::active();

  if (cmd == "ping") {
    return "{\"ok\":true,\"service\":\"EX-CAD\",\"port\":" +
           std::to_string(AiBridgeServer::instance().port()) + "}";
  }

  if (cmd == "sketch_new") {
    const std::string plane_str = json_extract_string(payload, "plane", "XY");
    SketchPlane plane = SketchPlane::XY;
    if (plane_str == "XZ") {
      plane = SketchPlane::XZ;
    }
    else if (plane_str == "YZ") {
      plane = SketchPlane::YZ;
    }
    doc.reset(plane);
    return "{\"ok\":true,\"plane\":\"" + plane_str + "\"}";
  }

  if (cmd == "sketch_add_rect") {
    const double u0 = json_extract_double(payload, "u0", -0.5);
    const double v0 = json_extract_double(payload, "v0", -0.5);
    const double w = json_extract_double(payload, "width", 1.0);
    const double h = json_extract_double(payload, "height", 1.0);
    doc.add_rect(u0, v0, w, h);
    return "{\"ok\":true,\"points\":" + std::to_string(doc.points.size()) + "}";
  }

  if (cmd == "sketch_add_circle") {
    const double cu = json_extract_double(payload, "cu", 0.0);
    const double cv = json_extract_double(payload, "cv", 0.0);
    const double r = json_extract_double(payload, "radius", 0.5);
    doc.add_circle(cu, cv, r);
    return "{\"ok\":true,\"entities\":" + std::to_string(doc.entities.size()) + "}";
  }

  if (cmd == "sketch_add_line") {
    const double u0 = json_extract_double(payload, "u0", 0.0);
    const double v0 = json_extract_double(payload, "v0", 0.0);
    const double u1 = json_extract_double(payload, "u1", 1.0);
    const double v1 = json_extract_double(payload, "v1", 0.0);
    doc.add_line(u0, v0, u1, v1);
    return "{\"ok\":true,\"entities\":" + std::to_string(doc.entities.size()) + "}";
  }

  if (cmd == "sketch_solve") {
    const SolveResult res = doc.solve();
    std::ostringstream oss;
    oss << "{\"ok\":true,\"converged\":" << (res.converged ? "true" : "false")
        << ",\"iterations\":" << res.iterations << ",\"max_residual\":" << res.max_residual << "}";
    return oss.str();
  }

  if (cmd == "sketch_extrude") {
    doc.solve();
    const float depth = static_cast<float>(json_extract_double(payload, "depth", 1.0));
    const bool symmetric = json_extract_bool(payload, "symmetric", false);
    const std::string name = json_extract_string(payload, "name", "EXCAD_Extrude");
    const RawMeshData raw = extrude_sketch(doc, depth, symmetric);
    if (raw.verts.empty()) {
      return "{\"ok\":false,\"error\":\"No closed sketch profile to extrude\"}";
    }
    create_mesh_object_from_raw(C, name.c_str(), raw);
    return "{\"ok\":true,\"verts\":" + std::to_string(raw.verts.size()) +
           ",\"faces\":" + std::to_string(raw.faces.size()) + "}";
  }

  if (cmd == "sketch_revolve") {
    doc.solve();
    const float angle = static_cast<float>(json_extract_double(payload, "angle_deg", 360.0));
    const int segs = json_extract_int(payload, "segments", 48);
    const std::string name = json_extract_string(payload, "name", "EXCAD_Revolve");
    const RawMeshData raw = revolve_sketch(doc, angle, segs);
    if (raw.verts.empty()) {
      return "{\"ok\":false,\"error\":\"No closed sketch profile to revolve\"}";
    }
    create_mesh_object_from_raw(C, name.c_str(), raw);
    return "{\"ok\":true,\"verts\":" + std::to_string(raw.verts.size()) +
           ",\"faces\":" + std::to_string(raw.faces.size()) + "}";
  }

  if (cmd == "sdf_humanoid") {
    HumanoidParams hp;
    hp.height = static_cast<float>(json_extract_double(payload, "height", 1.75));
    hp.muscularity = static_cast<float>(json_extract_double(payload, "muscularity", 0.5));
    hp.shoulder_ratio = static_cast<float>(json_extract_double(payload, "shoulder_ratio", 1.0));
    hp.hip_ratio = static_cast<float>(json_extract_double(payload, "hip_ratio", 1.0));
    hp.resolution = json_extract_int(payload, "resolution", 72);
    const std::string name = json_extract_string(payload, "name", "EXCAD_Humanoid");
    const RawMeshData raw = generate_humanoid_sdf(hp);
    create_mesh_object_from_raw(C, name.c_str(), raw);
    return "{\"ok\":true,\"verts\":" + std::to_string(raw.verts.size()) +
           ",\"faces\":" + std::to_string(raw.faces.size()) + "}";
  }

  if (cmd == "sdf_custom") {
    const std::string spec = json_extract_string(payload, "spec", "");
    const int res = json_extract_int(payload, "resolution", 64);
    const std::string name = json_extract_string(payload, "name", "EXCAD_SDF");
    const RawMeshData raw = generate_custom_sdf_from_spec(spec, res);
    create_mesh_object_from_raw(C, name.c_str(), raw);
    return "{\"ok\":true,\"verts\":" + std::to_string(raw.verts.size()) +
           ",\"faces\":" + std::to_string(raw.faces.size()) + "}";
  }

  return "{\"ok\":false,\"error\":\"Unknown command: " + cmd + "\"}";
}

}  // namespace

AiBridgeServer &AiBridgeServer::instance()
{
  static AiBridgeServer server;
  return server;
}

AiBridgeServer::~AiBridgeServer()
{
  stop();
}

bool AiBridgeServer::start(int port)
{
  if (running_.load()) {
    return true;
  }
  port_ = port;
  running_.store(true);
  accept_thread_ = std::thread(&AiBridgeServer::server_loop, this);
  return true;
}

void AiBridgeServer::stop()
{
  if (!running_.exchange(false)) {
    return;
  }
  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }
}

int AiBridgeServer::process_pending(bContext *C)
{
  int count = 0;
  for (;;) {
    PendingRequest req;
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (queue_.empty()) {
        break;
      }
      req = std::move(queue_.front());
      queue_.pop();
    }
    const std::string response = dispatch_command(C, req.payload);
    req.reply_fn(response);
    ++count;
  }
  return count;
}

void AiBridgeServer::server_loop()
{
#ifdef _WIN32
  WSADATA wsa_data{};
  if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
    running_.store(false);
    return;
  }
#endif

  excad_socket_t listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (listen_sock == kInvalidSocket) {
    running_.store(false);
#ifdef _WIN32
    WSACleanup();
#endif
    return;
  }

  int opt = 1;
  setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&opt), sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

  if (bind(listen_sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
      listen(listen_sock, 8) != 0)
  {
    close_socket(listen_sock);
    running_.store(false);
#ifdef _WIN32
    WSACleanup();
#endif
    return;
  }

  while (running_.load()) {
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(listen_sock, &read_fds);
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 100000;

    const int ready = select(static_cast<int>(listen_sock) + 1, &read_fds, nullptr, nullptr, &tv);
    if (ready <= 0) {
      continue;
    }

    excad_socket_t client = accept(listen_sock, nullptr, nullptr);
    if (client == kInvalidSocket) {
      continue;
    }

    std::string buffer;
    char chunk[4096];
    for (;;) {
      const int n = recv(client, chunk, sizeof(chunk), 0);
      if (n <= 0) {
        break;
      }
      buffer.append(chunk, static_cast<size_t>(n));
      if (buffer.find('\n') != std::string::npos) {
        break;
      }
    }

    if (!buffer.empty()) {
      auto prom = std::make_shared<std::promise<std::string>>();
      auto fut = prom->get_future();
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_.push(PendingRequest{
            buffer,
            [prom](const std::string &resp) { prom->set_value(resp); },
        });
      }
      std::string reply = "{\"ok\":false,\"error\":\"Timeout\"}";
      if (fut.wait_for(std::chrono::seconds(15)) == std::future_status::ready) {
        reply = fut.get();
      }
      reply.push_back('\n');
      send(client, reply.c_str(), static_cast<int>(reply.size()), 0);
    }

    close_socket(client);
  }

  close_socket(listen_sock);
#ifdef _WIN32
  WSACleanup();
#endif
}

}  // namespace blender::ed::excad
