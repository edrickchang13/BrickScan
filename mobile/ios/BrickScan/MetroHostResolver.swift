// MetroHostResolver.swift
// Port of the original BSResolveMetroHost (ObjC) from AppDelegate.mm.
//
// Purpose:
//   Enumerate all non-loopback IPv4 interfaces on the device — including
//   the point-to-point peer address on USB link-local (169.254.x.x) — probe
//   each concurrently for a live Metro bundler at http://<host>:8081/status,
//   and return the first one that answers.
//
// Drop this file into mobile/ios/BrickScan/ and add to the BrickScan target.
// Then in AppDelegate.swift's ReactNativeDelegate.bundleURL(), replace:
//     return RCTBundleURLProvider.sharedSettings().jsBundleURL(forBundleRoot: ".expo/.virtual-metro-entry")
// with:
//     let host = MetroHostResolver.resolve()
//     RCTBundleURLProvider.sharedSettings().jsLocation = host
//     return RCTBundleURLProvider.sharedSettings().jsBundleURL(forBundleRoot: ".expo/.virtual-metro-entry")

import Foundation
import Darwin
import React

#if DEBUG
enum MetroHostResolver {
  /// Public entry point — resolves the Metro host to use for this launch.
  ///   1. Scan all live interfaces concurrently for a live Metro bundler.
  ///   2. Fall back to saved jsLocation if it still answers.
  ///   3. Fall back to "localhost:8081" (simulator / loopback).
  static func resolve() -> String {
    if let scanned = scanForLiveHost() {
      return scanned
    }
    let saved = RCTBundleURLProvider.sharedSettings().jsLocation
    if let saved = saved, !saved.isEmpty, isMetroReachable(at: saved) {
      return saved
    }
    return "localhost:8081"
  }

  // MARK: - Interface enumeration

  /// Every non-loopback IPv4 on this device, plus P2P peer addresses
  /// (`ifa_dstaddr` on `IFF_POINTOPOINT` interfaces) which give the Mac's
  /// side of a USB link.
  static func allLocalIPv4Addresses() -> [String] {
    var addrs: [String] = []
    var ifap: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&ifap) == 0, let first = ifap else { return addrs }
    defer { freeifaddrs(ifap) }

    var cursor: UnsafeMutablePointer<ifaddrs>? = first
    while let ifa = cursor {
      let pointer = ifa.pointee
      let name = String(cString: pointer.ifa_name)
      let flags = pointer.ifa_flags

      // Own interface address
      if let sockaddr = pointer.ifa_addr, sockaddr.pointee.sa_family == UInt8(AF_INET) {
        if !name.hasPrefix("lo") {
          if let ip = ipv4String(fromSockaddr: sockaddr), !addrs.contains(ip) {
            addrs.append(ip)
          }
        }
      }

      // P2P peer address — Mac's side of USB link
      if (flags & UInt32(IFF_POINTOPOINT)) != 0,
         let dst = pointer.ifa_dstaddr,
         dst.pointee.sa_family == UInt8(AF_INET) {
        if let ip = ipv4String(fromSockaddr: dst),
           !ip.hasPrefix("127."),
           !addrs.contains(ip) {
          addrs.append(ip)
        }
      }

      cursor = pointer.ifa_next
    }
    return addrs
  }

  private static func ipv4String(fromSockaddr sa: UnsafeMutablePointer<sockaddr>) -> String? {
    var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
    let result = getnameinfo(
      sa, socklen_t(sa.pointee.sa_len),
      &host, socklen_t(host.count),
      nil, 0,
      NI_NUMERICHOST
    )
    guard result == 0 else { return nil }
    return String(cString: host)
  }

  // MARK: - Probing

  /// Probe Metro `/status` at the given host. Up to 2 attempts (3 s each),
  /// with 1 s pause between. `host` may be "ip" or "ip:port"; falls back to 8081.
  static func isMetroReachable(at host: String) -> Bool {
    let hasPort = host.contains(":")
    let urlString = hasPort ? "http://\(host)/status" : "http://\(host):8081/status"
    guard let url = URL(string: urlString) else { return false }

    for attempt in 0..<2 {
      if attempt > 0 { Thread.sleep(forTimeInterval: 1.0) }

      var req = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 3.0)
      req.httpMethod = "HEAD"

      let sem = DispatchSemaphore(value: 0)
      var ok = false
      let task = URLSession.shared.dataTask(with: req) { _, response, error in
        if error == nil, let http = response as? HTTPURLResponse, http.statusCode < 500 {
          ok = true
        }
        sem.signal()
      }
      task.resume()
      _ = sem.wait(timeout: .now() + 3.5)
      if ok { return true }
    }
    return false
  }

  /// Probe every interface IP concurrently; return the first that answers.
  static func scanForLiveHost() -> String? {
    let candidates = allLocalIPv4Addresses()
    guard !candidates.isEmpty else { return nil }

    let lock = NSLock()
    var found: String?
    let group = DispatchGroup()
    let queue = DispatchQueue.global(qos: .userInitiated)

    for ip in candidates {
      group.enter()
      queue.async {
        defer { group.leave() }
        if isMetroReachable(at: ip) {
          lock.lock()
          if found == nil { found = ip }
          lock.unlock()
        }
      }
    }

    _ = group.wait(timeout: .now() + 8.0)
    return found
  }
}
#endif
