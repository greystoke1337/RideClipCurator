"""Project-wide constants. Tune these during the pilot phase (spec §10.2)."""

from pathlib import Path

# --- proxy transcode (spec 5.1) ---
PROXY_HEIGHT = 720
PROXY_CODEC = "libx264"
PROXY_PRESET = "veryfast"
PROXY_CRF = 28

# --- two-camera sync (spec 5.2) ---
# Cameras' clocks drift. Start here, then validate against a couple of known
# matching moments and adjust CAMERA_OFFSET_SECONDS in sync.py's call site.
SYNC_TOLERANCE_SECONDS = 30

# --- dedup (spec 5.3, native Python stand-in for videoduplicatefinder) ---
DEDUP_SAMPLE_FRAMES = 5
DEDUP_HASH_SIZE = 16
DEDUP_MAX_HAMMING_DISTANCE = 12  # lower = stricter match

# Embedding-based near-duplicate detection: a pretrained CNN's pooled feature
# is a much better "practically identical" signal than perceptual hash alone
# (view/exposure-invariant), reusing torchvision rather than adding a new
# model dependency. Two clips cluster together if EITHER this or the phash
# distance above says they match.
DEDUP_EMBEDDING_MODEL = "mobilenet_v3_small"
DEDUP_EMBEDDING_SAMPLE_FRAMES = 5
DEDUP_EMBEDDING_SIMILARITY_THRESHOLD = 0.97  # cosine similarity, 1.0 = identical

# --- steadiness (spec 5.5) ---
MOTION_SAMPLE_FRAMES = 40

# --- direction of travel (forward/backward-facing mount) ---
# Ego-motion signal from the same optical flow steadiness already computes:
# a forward-facing camera's flow field expands radially outward from the
# center (approaching objects grow), a rear-facing one contracts inward.
# COHERENCE is how well the flow fits that radial model at all (0-1) — low
# coherence (panning, handheld, mostly-static shots) means the direction call
# isn't trustworthy and camera_direction should be "unclear" rather than guessed.
DIRECTION_MIN_COHERENCE = 0.35

# --- mount type: bike-mounted vs handheld ---
# Compound heuristic (like other_bike_visible): mounted footage has spatially
# coherent, mostly-radial flow (see above) despite high-frequency vibration;
# handheld has more rotational/erratic flow as the person's body and arm move.
# RAM tags are a strong secondary signal — indoor/body-part context almost
# always means handheld, mount-hardware context almost always means mounted.
MOUNT_COHERENCE_THRESHOLD = 0.35  # below this, motion alone suggests handheld
HANDHELD_TAG_WORDS = {
    "hand", "arm", "selfie", "room", "indoor", "hallway", "corridor",
    "stairs", "stairwell", "living room", "kitchen", "bedroom", "bathroom",
}
MOUNTED_TAG_WORDS = {"windshield", "dashboard", "handlebar", "mirror", "helmet"}

# --- audio (spec 5.6) ---
AUDIO_SILENCE_RMS_DBFS = -40.0  # below this, skip Whisper entirely
YAMNET_SPEECH_THRESHOLD = 0.3
YAMNET_SPEECH_CLASSES = {"Speech", "Conversation", "Narration, monologue"}

# --- content tags (spec 5.4) ---
TAG_SAMPLE_FRAMES = 5
MOTORCYCLE_TAG_WORDS = {"motorcycle", "motorbike", "moped", "biker", "motorcyclist"}

# --- color / golden hour (spec 5.8, optional) ---
COLOR_SAMPLE_FRAMES = 5
GOLDEN_HOUR_HUE_DEG = (10, 45)  # warm orange/gold band on OpenCV's 0-180 hue scale (x2 for degrees)
GOLDEN_HOUR_MIN_SATURATION = 60

# --- scoring (spec §7) — starting weights, expect to retune after the pilot ---
SCORE_WEIGHTS = {
    "steadiness": 0.30,
    "other_bike_visible": 0.30,
    "has_speech": 0.20,
    "landscape_variety": 0.15,
    "golden_hour": 0.05,
}

DB_FILENAME = "index.db"
THUMBNAIL_SUFFIX = "_thumb.jpg"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mts", ".m4v"}


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
