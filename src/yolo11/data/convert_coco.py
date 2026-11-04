
import json
import os
import shutil
import sys
from pathlib import Path

def convert_coco_json(json_file, output_dir):
    """
    Converts COCO JSON format to YOLO txt format with STRICT filtering.
    """
    save_dir = Path(output_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    # 1. Create Category Mapping (91 -> 80)
    # Confirmed: Sorting by ID aligns with official YOLO 80 classes.
    categories = sorted(data['categories'], key=lambda x: x['id'])
    coco_id_to_yolo = {cat['id']: idx for idx, cat in enumerate(categories)}
            
    # 2. Index Images
    images = {x['id']: x for x in data['images']}
    print(f"Total Unique Images: {len(images)}")
    
    # 3. Group Annotations by Image & Filter
    # Pre-scan annotations to filter duplicates and invalid ones
    img_to_anns = {}
    
    stats = {
        'total': 0,
        'iscrowd': 0,
        'invalid_dim': 0,
        'tiny': 0,
        'duplicates': 0,
        'valid': 0
    }
    
    seen_anns_per_img = {} # Key: (img_id, cat, x, y, w, h)
    
    for i, ann in enumerate(data['annotations']):
        if i % 50000 == 0:
            sys.stdout.write(f"\rFiltering annotations: {i}/{len(data['annotations'])}")
            sys.stdout.flush()
        stats['total'] += 1
        img_id = ann['image_id']
        
        # 1. IsCrowd Check
        if ann.get('iscrowd') == 1:
            stats['iscrowd'] += 1
            continue
            
        bbox = ann['bbox'] # x,y,w,h
        x, y, w, h = bbox
        
        # 2. Invalid Dimensions Check
        if w <= 0 or h <= 0:
            stats['invalid_dim'] += 1
            continue
            
        # 3. Tiny Object Check (< 1px area)
        if (w * h) < 1.0:
            stats['tiny'] += 1
            continue
            
        # 4. Duplicate Check
        # Round to 2 decimals for dup check to catch near-identical floats
        dup_key = (img_id, ann['category_id'], round(x,2), round(y,2), round(w,2), round(h,2))
        
        # Global set might be too big, keep per-image logic is safer but we iterate flat list.
        # Actually global set of tuples for 860k items is fine (memory wise ~100MB).
        if dup_key in seen_anns_per_img:
             stats['duplicates'] += 1
             continue
        seen_anns_per_img[dup_key] = True

        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        stats['valid'] += 1

    print("\n--- Filtering Stats ---")
    print(json.dumps(stats, indent=2))
    
    # 4. Write Files (Ensure ALL images get a file)
    processed_count = 0
    
    print("\nWriting files...")
    for i, (img_id, img_info) in enumerate(images.items()):
        if i % 10000 == 0:
            sys.stdout.write(f"\rWriting TXT: {i}/{len(images)}")
            sys.stdout.flush()
        processed_count += 1
        file_name = img_info['file_name'] 
        label_name = os.path.splitext(file_name)[0] + ".txt"
        label_path = save_dir / label_name
        
        img_w = img_info['width']
        img_h = img_info['height']
        
        anns = img_to_anns.get(img_id, [])
        lines = []
        
        for ann in anns:
            cls_idx = coco_id_to_yolo[ann['category_id']]
            x_min, y_min, w, h = ann['bbox']
            
            # Normalize with High Precision
            x_c = (x_min + w / 2) / img_w
            y_c = (y_min + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            # Clip rigorously
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            
            # Additional Sanity Check: if width or height becomes 0 after clip/norm, drop it
            if w_norm == 0 or h_norm == 0:
                continue
                
            lines.append(f"{cls_idx} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        # Write (creates empty file if lines is empty)
        with open(label_path, 'w') as f_out:
            if lines:
                f_out.write('\n'.join(lines) + '\n')
            else:
                pass # Empty file creation

    print(f"Stats: Processed {processed_count} images. Output directory: {save_dir}")

def verify_output(output_dir, expected_count, nc=80):
    print(f"\nVerifying output in {output_dir}...")
    files = list(Path(output_dir).glob("*.txt"))
    if len(files) != expected_count:
        print(f"CRITICAL: Expected {expected_count} files, found {len(files)}!")
    else:
        print(f"PASS: Count matches {expected_count}.")
        
    # Scan content for anomalies
    error_count = 0
    print("Scanning content...")
    for i, p in enumerate(files):
        if i % 10000 == 0:
             sys.stdout.write(f"\rScanning: {i}/{len(files)}")
             sys.stdout.flush()
        with open(p, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if len(parts) != 5:
                    print(f"Error format: {p}")
                    error_count += 1
                    break
                c, x, y, w, h = parts
                c = int(c)
                x, y, w, h = float(x), float(y), float(w), float(h)
                
                if c < 0 or c >= nc:
                    print(f"Error class {c} in {p}")
                    error_count += 1
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    print(f"Error bounds in {p}: {line}")
                    error_count += 1
                    
    if error_count == 0:
        print("PASS: Content verification successful. No format or value errors.")
    else:
        print(f"FAIL: Found {error_count} errors.")
