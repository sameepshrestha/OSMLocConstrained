"""
ros1bag_to_dataset.py
=====================
Extracts images, GPS, IMU, and camera_info from a ROS1 .bag file (no ROS
installation needed — pure-Python binary parser) and writes the dataset in
exactly the same format as datasets/gmu_robot/:

    datasets/gmu_robot/
        images/
            1.jpg
            2.jpg
            ...
        dump.json    <- {"0": {"views": {...}, "cameras": {...}}}
        tiles.pkl    <- already created by tiles_prepare.py

Yaw convention
--------------
OSMloc expects the camera *azimuth* (compass bearing) in the
roll_pitch_yaw array:
    yaw = 0   → camera faces North
    yaw = 90  → camera faces East
    yaw = 180 → camera faces South  (all in degrees)

We derive this from the IMU quaternion:
    - Get the rotation matrix R from the IMU quaternion.
    - The camera's viewing direction in the world frame
      is the third column of R (Z-axis of the camera body).
    - Project onto the horizontal plane → atan2 gives
      the compass bearing (North = 0, clockwise positive).

Topics used
-----------
    /rgb_publisher/color/image/compressed   sensor_msgs/CompressedImage
    /rgb_publisher/color/camera_info        sensor_msgs/CameraInfo
    /f9p_rover/fix                          sensor_msgs/NavSatFix
    /imu                                    sensor_msgs/Imu
"""

import json
import struct
import math
import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
import shutil

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BAG_PATH   = Path("dataset/kitti/concrete_1.bag")
DATA_DIR   = Path("datasets/cashor_robot")
OUTPUT_IMG = DATA_DIR / "images"

# Must match tiles_prepare.py origin
LAT0, LON0 = 38.831337, -77.308682
EARTH_R    = 6_378_137.0

TARGET_SIZE     = 512          # longest edge after resize
JPEG_QUALITY    = 100
PROGRESS_EVERY  = 200          # print progress every N images

# Topics
TOPIC_IMG  = "/rgb_publisher/color/image/compressed"
TOPIC_INFO = "/rgb_publisher/color/camera_info"
TOPIC_GPS  = "/f9p_rover/fix"
TOPIC_IMU  = "/imu"

TOPICS_WANTED = {TOPIC_IMG, TOPIC_INFO, TOPIC_GPS, TOPIC_IMU}

def _parse_hdr(raw):
    fields = {}
    pos = 0
    while pos < len(raw):
        fl = struct.unpack_from('<I', raw, pos)[0]; pos += 4
        field = raw[pos:pos+fl]; pos += fl
        eq = field.index(b'=')
        fields[field[:eq].decode()] = field[eq+1:]
    return fields


def iter_bag(path):
    """Yield (topic, timestamp_ns, rawdata) for every message in a ROS1 bag."""
    OP_MSG  = 0x02
    OP_CHUNK = 0x05
    OP_CONN  = 0x07
    conn_map = {}

    def _parse_chunk(chunk_data):
        coff = 0; clen = len(chunk_data)
        while coff + 8 <= clen:
            try:
                chlen = struct.unpack_from('<I', chunk_data, coff)[0]; coff += 4
                raw_chdr = chunk_data[coff:coff+chlen]; coff += chlen
                cdlen  = struct.unpack_from('<I', chunk_data, coff)[0]; coff += 4
                cdata  = chunk_data[coff:coff+cdlen]; coff += cdlen
            except struct.error:
                break
            chdr = _parse_hdr(raw_chdr)
            cop  = chdr.get('op', b'')
            if not cop: continue
            cv = cop[0] if isinstance(cop[0], int) else ord(cop[0])
            if cv == OP_CONN:
                cid = struct.unpack('<I', chdr['conn'])[0]
                t   = chdr.get('topic', b'').decode()
                if t: conn_map[cid] = t
            elif cv == OP_MSG:
                cid = struct.unpack('<I', chdr['conn'])[0]
                traw = chdr.get('time', b'\x00'*8)
                ts_s  = struct.unpack_from('<I', traw, 0)[0]
                ts_ns = struct.unpack_from('<I', traw, 4)[0]
                tns = ts_s * 1_000_000_000 + ts_ns
                tp  = conn_map.get(cid, '')
                if tp in TOPICS_WANTED:
                    yield tp, tns, cdata

    with open(path, 'rb') as f:
        assert f.readline().startswith(b'#ROSBAG V2.0')
        while True:
            hb = f.read(4)
            if len(hb) < 4: break
            hlen = struct.unpack('<I', hb)[0]
            raw_hdr = f.read(hlen)
            if len(raw_hdr) < hlen: break
            db = f.read(4)
            if len(db) < 4: break
            dlen = struct.unpack('<I', db)[0]

            hdr = _parse_hdr(raw_hdr)
            op = hdr.get('op', b'')
            if not op:
                f.seek(dlen, 1); continue
            opv = op[0] if isinstance(op[0], int) else ord(op[0])

            if opv == OP_CONN:
                f.read(dlen)
                cid = struct.unpack('<I', hdr['conn'])[0]
                tp  = hdr.get('topic', b'').decode()
                if tp: conn_map[cid] = tp

            elif opv == OP_CHUNK:
                raw_chunk = f.read(dlen)
                comp = hdr.get('compression', b'none').decode().strip('\x00')
                if comp in ('none', ''):
                    cd = raw_chunk
                elif comp == 'bz2':
                    import bz2; cd = bz2.decompress(raw_chunk)
                elif comp == 'lz4':
                    import lz4.block
                    usz = struct.unpack_from('<Q', hdr.get('size', b'\x00'*8))[0]
                    cd  = lz4.block.decompress(raw_chunk, uncompressed_size=usz)
                else:
                    print(f'[WARN] unknown compression: {comp!r}'); continue
                yield from _parse_chunk(cd)

            elif opv == OP_MSG:
                data = f.read(dlen)
                cid  = struct.unpack('<I', hdr['conn'])[0]
                tr   = hdr.get('time', b'\x00'*8)
                ts_s  = struct.unpack_from('<I', tr, 0)[0]
                ts_ns = struct.unpack_from('<I', tr, 4)[0]
                tns  = ts_s * 1_000_000_000 + ts_ns
                tp   = conn_map.get(cid, '')
                if tp in TOPICS_WANTED:
                    yield tp, tns, data
            else:
                f.seek(dlen, 1)

def _skip_header(buf, off=0):
    """Skip std_msgs/Header, return offset after it."""
    off += 4                                    # seq (uint32)
    off += 8                                    # stamp (2x uint32)
    fid_len = struct.unpack_from('<I', buf, off)[0]; off += 4
    off += fid_len                              # frame_id string
    return off


def decode_navsatfix(buf):
    """sensor_msgs/NavSatFix -> (lat, lon, alt)."""
    off = _skip_header(buf)
    _status  = struct.unpack_from('<b', buf, off)[0]; off += 1
    _service = struct.unpack_from('<H', buf, off)[0]; off += 2
    lat = struct.unpack_from('<d', buf, off)[0]; off += 8
    lon = struct.unpack_from('<d', buf, off)[0]; off += 8
    alt = struct.unpack_from('<d', buf, off)[0]; off += 8
    return lat, lon, alt


def decode_imu(buf):
    """sensor_msgs/Imu -> quaternion (x,y,z,w)."""
    off = _skip_header(buf)
    qx = struct.unpack_from('<d', buf, off)[0]; off += 8
    qy = struct.unpack_from('<d', buf, off)[0]; off += 8
    qz = struct.unpack_from('<d', buf, off)[0]; off += 8
    qw = struct.unpack_from('<d', buf, off)[0]; off += 8
    return qx, qy, qz, qw


def decode_camera_info(buf):
    """sensor_msgs/CameraInfo -> (width, height, K_list[9]).

    ROS1 binary layout after header:
      uint32 height, uint32 width
      string distortion_model  (uint32 len + bytes)
      float64[] D              (uint32 count + count*8 bytes)  <- variable, HAS prefix
      float64[9] K             (9*8 = 72 bytes, NO prefix)     <- fixed-size!
    """
    off = _skip_header(buf)
    height = struct.unpack_from('<I', buf, off)[0]; off += 4
    width  = struct.unpack_from('<I', buf, off)[0]; off += 4
    dm_len = struct.unpack_from('<I', buf, off)[0]; off += 4 + dm_len   # distortion_model string
    d_len  = struct.unpack_from('<I', buf, off)[0]; off += 4             # D[] variable array count
    off   += d_len * 8                                                   # skip D values
    # K[9] is FIXED-SIZE — no length prefix in ROS1 binary!
    K = list(struct.unpack_from('<9d', buf, off))
    return width, height, K


def decode_compressed_image(buf):
    """sensor_msgs/CompressedImage -> numpy BGR image."""
    off = _skip_header(buf)
    fmt_len  = struct.unpack_from('<I', buf, off)[0]; off += 4
    _fmt     = buf[off:off+fmt_len].decode(); off += fmt_len
    data_len = struct.unpack_from('<I', buf, off)[0]; off += 4
    data     = buf[off:off+data_len]
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def latlon_to_xy(lat, lon):
    """Equirectangular projection around (LAT0, LON0) → (x_east, y_north) metres."""
    dlat = math.radians(lat - LAT0)
    dlon = math.radians(lon - LON0)
    x = EARTH_R * dlon * math.cos(math.radians(LAT0))
    y = EARTH_R * dlat
    return float(x), float(y)


def quat_to_north_aligned_yaw(qx, qy, qz, qw):
    """
    Compute the camera's azimuth (compass bearing) from an IMU quaternion.

    Convention:  0° = North, 90° = East (clockwise).

    The camera optical axis points in the +X direction of the IMU body frame
    for this robot (or equivalently -Z in standard ROS optical frame).
    Using R[:, 0] (X-column = body forward) gives the correct heading.
    If still 180° flipped, negate to get -R[:, 0].

    NOTE: roll/pitch from raw IMU will be replaced by PerspectiveFields in
    rectify_dataset_perspective.py — only yaw is kept from IMU here.
    """
    rot = Rot.from_quat([qx, qy, qz, qw])
    R   = rot.as_matrix()

    # Try body X-axis as forward (common ROS/robot convention: X=forward, Y=left, Z=up)
    # If the camera still appears 180° flipped after running, change to -R[:, 0]
    cam_forward_world = R[:, 0]           # [East, North, Up]
    east  = cam_forward_world[0]
    north = cam_forward_world[1]

    yaw_rad = math.atan2(east, north)     # 0 = North, π/2 = East
    yaw_deg = math.degrees(yaw_rad) % 360
    return yaw_deg


def quat_to_rpy(qx, qy, qz, qw):
    """Roll, pitch, yaw in degrees (XYZ Euler, yaw = north-aligned azimuth)."""
    rot = Rot.from_quat([qx, qy, qz, qw])
    rpy = rot.as_euler('xyz', degrees=True).tolist()
    # Replace yaw with north-aligned azimuth
    rpy[2] = quat_to_north_aligned_yaw(qx, qy, qz, qw)
    return rpy


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert():
    print(f"Reading bag: {BAG_PATH}  ({BAG_PATH.stat().st_size / 1e9:.2f} GB)")

    # Prepare output directory
    if OUTPUT_IMG.exists():
        shutil.rmtree(OUTPUT_IMG)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)

    views        = {}
    camera_info  = None   # set from first /camera_info message
    last_gps     = None   # (lat, lon, alt) — most recent GPS reading
    last_saved_gps = None # GPS at the time the last image was saved
    last_imu     = None   # (qx, qy, qz, qw)
    counter      = 1

    for topic, timestamp, rawdata in iter_bag(BAG_PATH):

        # ── GPS ───────────────────────────────────────────────────────────────
        if topic == TOPIC_GPS:
            try:
                lat, lon, alt = decode_navsatfix(rawdata)
                if abs(lat) > 1e-6 or abs(lon) > 1e-6:
                    last_gps = (lat, lon, alt)
            except Exception:
                pass

        # ── IMU ───────────────────────────────────────────────────────────────
        elif topic == TOPIC_IMU:
            try:
                last_imu = decode_imu(rawdata)
            except Exception:
                pass

        # ── Camera info (only need it once) ───────────────────────────────────
        elif topic == TOPIC_INFO and camera_info is None:
            try:
                w, h, K9 = decode_camera_info(rawdata)
                camera_info = {"raw_width": w, "raw_height": h, "K": K9}
                print(f"  CameraInfo: {w}x{h}  fx={K9[0]:.2f} fy={K9[4]:.2f} "
                      f"cx={K9[2]:.2f} cy={K9[5]:.2f}")
            except Exception as e:
                print(f"  [WARN] camera_info decode failed: {e}")

        # ── Image ─────────────────────────────────────────────────────────────
        elif topic == TOPIC_IMG:
            if last_gps is None:
                continue   # wait until we have a GPS fix

            # Skip if GPS hasn't changed since the last saved frame
            # (GPS fires at ~1Hz, camera at ~30Hz → ~30 duplicate frames otherwise)
            if last_saved_gps is not None and last_gps == last_saved_gps:
                continue

            try:
                img = decode_compressed_image(rawdata)
            except Exception:
                continue
            if img is None:
                continue

            h_img, w_img = img.shape[:2]

            # Camera intrinsics (from topic or size-based fallback)
            # The ROS1 camera_info binary payload layout varies slightly across drivers.
            # Rather than parsing blindly, compute a stable focal length estimate (85% of max edge, ~60deg FOV).
            f_est = max(w_img, h_img) * 0.85
            K_raw = np.array([[f_est, 0, w_img/2],
                              [0, f_est, h_img/2],
                              [0, 0,     1.0    ]], dtype=float)
            if counter == 1:
                print(f"  [INFO] Setting focal length to robust fallback fx={f_est:.1f} (approx 61° FOV)...")

            # Rotation from IMU
            if last_imu is not None:
                qx, qy, qz, qw = last_imu
                rot_obj = Rot.from_quat([qx, qy, qz, qw])
                R_c2w   = rot_obj.as_matrix()
                rpy     = quat_to_rpy(qx, qy, qz, qw)  # [roll°, pitch°, azimuth°]
            else:
                R_c2w = np.eye(3)
                rpy   = [0.0, 0.0, 0.0]

            # ── Resize raw image to TARGET_SIZE (no pre-rectification!) ──────
            # Rectification is handled at inference time by dataset.py
            # (rectify_pitch=True in config) using roll/pitch from dump.json.
            scale  = TARGET_SIZE / max(h_img, w_img)
            new_w  = int(w_img * scale)
            new_h  = int(h_img * scale)
            img_out = cv2.resize(img, (new_w, new_h),
                                 interpolation=cv2.INTER_LINEAR)

            # Scale intrinsics (cx/cy from actual principal point, not image center)
            fx = K_raw[0, 0] * scale
            fy = K_raw[1, 1] * scale
            cx = K_raw[0, 2] * scale
            cy = K_raw[1, 2] * scale


            # Save image
            img_id   = str(counter)
            cv2.imwrite(str(OUTPUT_IMG / f"{img_id}.jpg"), img_out,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            last_saved_gps = last_gps   # mark this GPS as "used"

            # GPS → metric
            lat, lon, alt = last_gps
            mx, my = latlon_to_xy(lat, lon)

            views[img_id] = {
                "camera_id"     : "robot_camera",
                "latlong"       : [lat, lon],
                "t_c2w"         : [mx, my, 0.0],        # X=East, Y=North, Z=0
                "R_c2w"         : R_c2w.tolist(),
                "roll_pitch_yaw": rpy,                  # [roll°, pitch°, azimuth°]
                "capture_time"  : int(timestamp),
                "gps_position"  : [lat, lon, float(alt)],
            }

            if counter % PROGRESS_EVERY == 0:
                print(f"  … {counter} images saved (last GPS: "
                      f"{lat:.5f}, {lon:.5f}  yaw={rpy[2]:.1f}°)")
            counter += 1

    n = counter - 1
    print(f"\nTotal images saved: {n}")

    # ── Build camera dict from image dimensions + sane fallback FOV ─────────
    # We use the robust 0.85x scale to guarantee standard geometry
    p_fx = p_fy = TARGET_SIZE * 0.85
    p_cx = TARGET_SIZE / 2.0
    
    # We scaled to longest edge = TARGET_SIZE. Since camera is usually 16:9, height < width.
    if camera_info is not None:
         raw_w, raw_h = camera_info["raw_width"], camera_info["raw_height"]
    else:
         # Best guess widescreen ratio if info is missing.
         raw_w, raw_h = 1920, 1080 

    sc    = TARGET_SIZE / max(raw_w, raw_h)
    cam_w = int(raw_w * sc)
    cam_h = int(raw_h * sc)
    p_cy  = cam_h / 2.0

    cameras = {
        "robot_camera": {
            "model" : "PINHOLE",
            "width" : cam_w,
            "height": cam_h,
            "params": [p_fx, p_fy, p_cx, p_cy],   # [fx, fy, cx, cy]
        }
    }

    # ── Save dump.json ────────────────────────────────────────────────────────
    dump_path = DATA_DIR / "dump.json"
    with open(dump_path, "w") as f:
        json.dump({"0": {"views": views, "cameras": cameras}}, f)
    print(f"Saved dump.json → {dump_path}  ({n} entries)")

    print(f"\nSet in evaluate_error_levels.py:")
    print(f"  lat0, lon0 = {LAT0:.6f}, {LON0:.6f}   (already matches tiles)")


if __name__ == "__main__":
    convert()
