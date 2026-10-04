# OpenXR baseline preparation — 03/10/2026

Goal: Windows D3D12 OpenXR sample -> Proton11 Experimental -> native WiVRn26.9 -> Quest3. No neural DLL/ReShade in this sample. Quest3 connected wirelessly on04/10/2026 through WiVRn26.9 Flatpak. Windows D3D12 sample created an OpenXR session and repeatedly submitted frames; logs identify Meta Quest3 on WiVRn, with server NVENC AV1 10-bit configuration. User confirmed visible VR scene, correct head tracking and working controllers. Basic headset baseline PASSED; neural stereo integration/performance remain untested. Initial launch bounded to120seconds. No neural rendering in this test.

## Prepared locally

- Official Khronos `hello_xr.exe` and loader1.1.63 (x64). Windows release ZIP SHA256 checked against GitHub release digest:01c631aeabbfe0879540f77ef833416c532a20746285b494630160c23588b771.
- Fedora44 WiVRn26.9-2 RPM unpacked here, not installed; Fedora signature/digests verified; no missing linked dependencies.
- WiVRn26.9 Quest client APK downloaded from official GitHub release, digest checked:1df6649ec77224fcc821af0ab4897222bdf3d7eb6ce6ad636461336724111331. NOT installed on headset.
- xrizer0.5 official release, digest checked:935ee21992d5cb99a2bce5b56014cc4d1f981bd9023f0eee65809b251ba7b1ee. Optional compatibility helper for Proton's legacy OpenVR bootstrap; no SteamVR installation. Runtime manifest/OpenVR paths are local. Bootstrap remains unverified with a headset.
- `prefix` is a separate copy of the desktop test prefix for existing Microsoft runtime dependencies. It is NOT a security sandbox. No experimental neural library is loaded.

## Checks performed

- Existing Proton11 contains wineopenxr and explicitly maps XR_KHR_D3D12_enable to XR_KHR_vulkan_enable. This is implemented support, not proof of headset compatibility or performance.
- Official Windows sample runs under Steam Linux Runtime/Proton; `--help` lists D3D12 and Stereo (saved logs/sample-help-windows.txt).
- Native loader successfully enumerates61 WiVRn extensions through local runtime.json. This loads the runtime library but does NOT exercise compositor, rendering or encoding.
- Without server/headset, Windows sample fails runtime negotiation with XR_ERROR_RUNTIME_UNAVAILABLE. Proton trace shows its Software/Wine/VR bootstrap key absent. Upstream wineopenxr still gates startup through OpenVR initialization; local xrizer override prepared, but it cannot complete initialization without a headset/runtime system. DO NOT mark this test a pass or assume headset attachment alone fixes it.
- WiVRn --help and wivrnctl --help work. No server has been left running. Launch scripts bash syntax checked.
- Independent read-only review found missing SteamGameId prevented Proton diagnostic logs. Fixed using test-specific424242 and OpenXR/VR tracing; rerun confirmed logs/steam-424242.log and acknowledged VR initialization failure. Windows cmd exits0 even when the GUI sample logs an error: judge the sample log/session, not shell exit status.

## Next interactive test

1. Power on Quest3 and connect Meta USB cable. Developer mode and USB debugging authorisation are required for USB streaming/sideloading; confirm device first. Existing adb: /home/michael/Projects/sdk/Android/Sdk/platform-tools/adb. Do not install APK onto an arbitrary connected phone: explicitly identify Quest serial/model first.
2. Install downloads/WiVRn-26.9.apk onto identified Quest only, with explicit adb serial.
3. Launch WiVRn server with local runtime/library paths; inspect logs and pairing PIN. Use authenticated pairing, not --no-encryption or --no-PIN. Upstream supports tcp_only and no-avahi for USB-only setup. Verify locally unpacked server can locate required resources first; fall back to supported native Fedora installation/dashboard if relocation causes errors.
4. USB protocol from upstream26.9 dashboard/adb.cpp: `adb -s QUEST_SERIAL reverse tcp:9757 tcp:9757`, then launch org.meumeu.wivrn through ACTION_VIEW with `wivrn+tcp://:PAIRING_PIN@127.0.0.1:9757`. Handle PIN privately, not in committed files. Disable headset Wi-Fi temporarily to ensure the test uses USB.
5. Once WiVRn reports connected, run `bash vr/run-sample.sh` from testbed root. Logs/headset-sample.txt captures Windows output. Local manifest and OpenVR override apply only to this process. Runtime-sharing needs PRESSURE_VESSEL_IMPORT_OPENXR_1_RUNTIMES=1 (included). xrizer exists for Proton initialization, not to translate the sample itself, which already uses OpenXR.
6. Pass requires reported WiVRn runtime, XR session creation with D3D12, visible stereoscopic cube scene, correct head/controller tracking and confirmed NVIDIA encoding. If bootstrap still fails, diagnose before any neural integration.

Sources:
- https://github.com/WiVRn/WiVRn/tree/v26.9
- https://github.com/WiVRn/WiVRn/blob/v26.9/docs/steamvr.md
- https://github.com/WiVRn/WiVRn/blob/v26.9/dashboard/adb.cpp
- https://github.com/ValveSoftware/Proton/tree/proton_11.0/wineopenxr
- https://github.com/KhronosGroup/OpenXR-SDK-Source/releases/tag/release-1.1.63

No games, host runtime selection, drivers, system packages, firewall rules or headset contents modified. Native WiVRn/NVIDIA motion-latency caveat still applies. Full stereo/neural experiments remain separate.
