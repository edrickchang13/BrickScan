internal import Expo
import React
import ReactAppDependencyProvider

@main
class AppDelegate: ExpoAppDelegate {
  var window: UIWindow?

  var reactNativeDelegate: ExpoReactNativeFactoryDelegate?
  var reactNativeFactory: RCTReactNativeFactory?

  public override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
  ) -> Bool {

#if DEBUG
    // ─── Evict stale Metro URLs on every cold launch ─────────────────────
    // expo-dev-client persists the last-used dev server URL in NSUserDefaults
    // under its own keys (separate from RCTBundleURLProvider.jsLocation).
    // USB link-local IPs change on every re-plug, so the cached URL goes
    // stale and the launcher shows "Could not connect to development server"
    // with an old IP. We clear every known dev-launcher / RN dev-settings
    // key so that MetroHostResolver's runtime scan is the only source of
    // truth for finding Metro.
    let defaults = UserDefaults.standard
    let stalePrefixes = [
      "EXDevLauncher",            // expo-dev-client's saved dev server URL + history
      "EXDevMenu",                // expo-dev-menu preferences
      "RCTDevMenu",               // RN core dev menu state
      "RCT_jsLocation",           // alt RN bundler location
      "RCTDevSettings",           // RN dev toggles (some store URLs)
    ]
    for key in defaults.dictionaryRepresentation().keys {
      if stalePrefixes.contains(where: { key.hasPrefix($0) }) {
        defaults.removeObject(forKey: key)
      }
    }
    // Also null the RCTBundleURLProvider's saved location so it doesn't
    // beat our resolver.
    RCTBundleURLProvider.sharedSettings().jsLocation = nil
    defaults.synchronize()
#endif

    let delegate = ReactNativeDelegate()
    let factory = ExpoReactNativeFactory(delegate: delegate)
    delegate.dependencyProvider = RCTAppDependencyProvider()

    reactNativeDelegate = delegate
    reactNativeFactory = factory

    window = UIWindow(frame: UIScreen.main.bounds)
    factory.startReactNative(
      withModuleName: "main",
      in: window,
      launchOptions: launchOptions)

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  public override func application(
    _ app: UIApplication,
    open url: URL,
    options: [UIApplication.OpenURLOptionsKey: Any] = [:]
  ) -> Bool {
    return super.application(app, open: url, options: options) || RCTLinkingManager.application(app, open: url, options: options)
  }

  public override func application(
    _ application: UIApplication,
    continue userActivity: NSUserActivity,
    restorationHandler: @escaping ([UIUserActivityRestoring]?) -> Void
  ) -> Bool {
    let result = RCTLinkingManager.application(application, continue: userActivity, restorationHandler: restorationHandler)
    return super.application(application, continue: userActivity, restorationHandler: restorationHandler) || result
  }
}

class ReactNativeDelegate: ExpoReactNativeFactoryDelegate {
  override func sourceURL(for bridge: RCTBridge) -> URL? {
    bridge.bundleURL ?? bundleURL()
  }

  override func bundleURL() -> URL? {
#if DEBUG
    // Auto-discover Metro host: scans all local interfaces (WiFi, USB link-local)
    // concurrently and picks whichever one answers at /status first. Falls back
    // to saved jsLocation, then localhost. See MetroHostResolver.swift.
    let host = MetroHostResolver.resolve()
    RCTBundleURLProvider.sharedSettings().jsLocation = host
    return RCTBundleURLProvider.sharedSettings().jsBundleURL(forBundleRoot: ".expo/.virtual-metro-entry")
#else
    return Bundle.main.url(forResource: "main", withExtension: "jsbundle")
#endif
  }
}
