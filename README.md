# ClassPulse — Multimodal Attendance & Engagement Agent

A local-first prototype for classroom attendance and session-level visual activity summaries. Built with Flask, OpenCV and a responsive dashboard.

## Features
- Student roster stored locally
- Webcam face sample capture and local LBPH face recognition
- Attendance marking once per recognized student per session
- Live face count, image motion and face-area indicators
- Session event logs and CSV attendance export
- Video-upload analysis for sampled face count and motion
- WAV audio signal-energy analysis with windowed RMS activity summary
- Rule-based teaching prompts
- Local data storage; no cloud account/API key required

## Quick start (Windows)
1. Install Python 3.10 or newer from https://www.python.org/downloads/ and enable **Add Python to PATH**.
2. Extract this ZIP to a folder.
3. Double-click `run_windows.bat`. The first run installs dependencies.
4. Open http://127.0.0.1:5000 in your browser.
5. Add each student, click **Capture face** for each one (the webcam opens), and ensure the student looks at the camera with varied but clear angles.
6. Click **Train / refresh face recognition model**.
7. Click **Start session** and allow camera access if requested.
8. Click **Stop session** when finished, then export attendance CSV.

## macOS / Linux
Run `chmod +x run_mac_linux.sh`, then `./run_mac_linux.sh`, and open http://127.0.0.1:5000.

## GitHub setup
1. Create a new repository on GitHub named `multimodal-attendance-engagement-agent`.
2. Keep personal face images and attendance records out of GitHub. The included `.gitignore` excludes local data.
3. In this project folder, run:
   ```bash
   git init
   git add .
   git commit -m "Initial ClassPulse attendance and engagement agent"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/multimodal-attendance-engagement-agent.git
   git push -u origin main
   ```
4. Replace `YOUR-USERNAME` with your GitHub username.

## Demo link
The local demo is `http://127.0.0.1:5000`, available only on the computer running the app. For a public mentor-review link, deploy it to a suitable host after testing and configure camera access/permissions. Webcam access on remote deployments can be restricted by browser security rules; local webcam use is the simplest demo.

## Limitations and responsible use
- This is a prototype, not a production biometric system. LBPH can misidentify people under poor lighting, camera angles, or similar appearances. Verify attendance manually.
- Face detection, movement and face-area ratio are not reliable measurements of attention, emotion, comprehension or learning.
- The dashboard's suggestions are simple heuristics, not an LLM or autonomous agent. Audio analysis measures signal energy only; it does not transcribe speech or identify speakers.
- Obtain informed consent, define retention/deletion rules, restrict access, and offer an equivalent non-camera option. Do not use automated outputs for grading, discipline, or high-impact decisions.
- The app binds to `127.0.0.1` by default and is intended for local use. Do not expose it publicly without authentication, HTTPS, access controls and security review.
- For privacy, the project stores captured face samples in `data/faces/` on the local computer. Delete that folder and local CSV/JSON files when no longer needed.
