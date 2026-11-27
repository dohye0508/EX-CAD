"""Frame extraction from a user-supplied video (requires ffmpeg on PATH).

Frames are written to a temporary folder for the LLM to inspect (viewpoints, sculpting order,
brush direction).  Nothing here claims to have 'watched' a video it could not open: failures are
returned as errors.
"""
import os
import shutil
import subprocess
import tempfile


def extract_frames(video_path, fps=1.0, max_frames=60, out_dir=None):
    if not os.path.exists(video_path):
        return {"ok": False, "error": f"video not found: {video_path}"}
    out_dir = out_dir or tempfile.mkdtemp(prefix="bnm_frames_")
    ff = shutil.which("ffmpeg")
    if ff:
        cmd = [ff, "-loglevel", "error", "-i", video_path, "-vf", f"fps={fps}", "-frames:v", str(max_frames),
               os.path.join(out_dir, "frame_%04d.png")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr.strip()[:500]}
    else:
        return {"ok": False, "error": "ffmpeg not found on PATH; install ffmpeg to analyse videos"}
    frames = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png"))
    return {"ok": True, "frames": frames, "dir": out_dir, "note": "temporary; delete with cleanup(dir)"}


def cleanup(directory):
    if directory and os.path.basename(directory).startswith("bnm_frames_"):
        shutil.rmtree(directory, ignore_errors=True)


# Template the LLM fills while watching a professional's sculpting video; stored in knowledge/.
WORKFLOW_OBSERVATION_TEMPLATE = {
    "source": "", "accessible": True, "initial_mesh": "", "phases": [
        {"order": 1, "goal": "", "brushes": [], "brush_direction": "", "radius_relative_to_head": None,
         "strength": "", "subdivision_level_or_density": "", "checks_from_views": [], "notes": ""}],
    "general_rules_extracted": [],
}
