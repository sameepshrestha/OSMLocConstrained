"""Parse a ROS1 .bag file header to extract topic/message-type info.
No ROS installation needed — pure Python binary parsing.
"""
import struct
import sys
from pathlib import Path
from collections import defaultdict

BAG_PATH = Path("./dataset/kitti/Nov4_C5.bag")

def read_header_record(f):
    """Read one header record and return dict of fields."""
    header_len = struct.unpack('<I', f.read(4))[0]
    raw = f.read(header_len)
    fields = {}
    pos = 0
    while pos < len(raw):
        field_len = struct.unpack('<I', raw[pos:pos+4])[0]
        pos += 4
        field = raw[pos:pos+field_len]
        pos += field_len
        eq = field.index(b'=')
        key = field[:eq].decode('utf-8')
        fields[key] = field[eq+1:]
    return fields

def inspect_bag(path):
    topic_counts = defaultdict(int)
    topic_types = {}

    with open(path, 'rb') as f:
        # Check magic
        magic = f.readline()
        if not magic.startswith(b'#ROSBAG V2.0'):
            print(f"Not a ROS1 bag? Magic: {magic}")
            return

        while True:
            header_len_bytes = f.read(4)
            if len(header_len_bytes) < 4:
                break
            header_len = struct.unpack('<I', header_len_bytes)[0]
            raw_header = f.read(header_len)
            data_len = struct.unpack('<I', f.read(4))[0]
            f.read(data_len)  # skip data

            # Parse header fields
            fields = {}
            pos = 0
            while pos < len(raw_header):
                fl = struct.unpack('<I', raw_header[pos:pos+4])[0]
                pos += 4
                field = raw_header[pos:pos+fl]
                pos += fl
                eq = field.index(b'=')
                key = field[:eq].decode('utf-8')
                fields[key] = field[eq+1:]

            op = fields.get('op', b'')
            if len(op) == 1:
                op_val = op[0]
            else:
                continue

            # op=0x02 -> Connection record (has topic + type)
            if op_val == 0x07:  # connection
                topic = fields.get('topic', b'').decode('utf-8')
                msg_type = fields.get('type', b'').decode('utf-8')
                if topic:
                    topic_types[topic] = msg_type
            # op=0x02 -> message data
            elif op_val == 0x02:
                conn_id = struct.unpack('<I', fields.get('conn', b'\x00\x00\x00\x00'))[0]
                # We'll count by conn id; topic mapping done above

    print("Topics found in bag:")
    print("="*60)
    for topic, mtype in sorted(topic_types.items()):
        print(f"  {topic}")
        print(f"      type: {mtype}")
    print(f"\nTotal unique topics: {len(topic_types)}")

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else BAG_PATH
    inspect_bag(path)
