import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import re
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from faster_whisper import WhisperModel
import ctranslate2
import sentencepiece
import traceback

# ======================== CẤU HÌNH ========================
MODELS_FOLDER = r"D:\VietSub_Tools\models"
CT2_MODELS_FOLDER = r"D:\VietSub_Tools\models\ctranslate2"
LOG_DIR = r"D:\VietSub_Tools\logs"
BATCH_SIZE = 8
VIDEO_EXTS = [".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".ts", ".m2ts"]
SUBTITLE_EXTS = [".srt", ".vtt", ".ass"]
MAX_SUB_LENGTH = 40

os.makedirs(LOG_DIR, exist_ok=True)

batch_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"{batch_time}.txt")

translator = None
sp_src = None
sp_tgt = None
current_lang = None

# ======================== HÀM TIỆN ÍCH ========================
def log_msg(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_line = f"[{timestamp}] [{level:7s}] {msg}"
    print(log_line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

def format_time(seconds):
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"

def detect_lang_regex(text):
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
    if re.search(r'[\u3040-\u30ff]', text):
        return "ja"
    if re.search(r'[\u0400-\u04FF]', text):
        return "ru"
    return "en"

def read_srt(file_path):
    subs = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            idx, time, *text = lines
            subs.append({"index": idx, "time": time, "text": "\n".join(text)})
    return subs

def write_srt(file_path, subs):
    with open(file_path, "w", encoding="utf-8") as f:
        for sub in subs:
            f.write(f"{sub['index']}\n{sub['time']}\n{sub['text']}\n\n")

def split_long_subs(subs, max_length=40):
    result = []
    idx_counter = 1
    for sub in subs:
        text = sub["text"]
        if len(text) <= max_length:
            result.append(sub)
            idx_counter += 1
        else:
            parts = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            for part in parts:
                result.append({
                    "index": str(idx_counter),
                    "time": sub["time"],
                    "text": part
                })
                idx_counter += 1
    return result

def extract_audio(video_path, audio_path, apply_denoise=False):
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"
        ]
        
        if apply_denoise:
            log_msg(f"    Áp dụng denoise filter", "DEBUG")
            cmd.extend(["-af", "highpass=f=100,lowpass=f=8000"])
        
        cmd.append(audio_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
            raise Exception(f"Exit code {result.returncode}: {error_msg[:100]}")
        
        if not os.path.exists(audio_path):
            raise Exception("Audio file not created")
        
        audio_size = os.path.getsize(audio_path) / 1024 / 1024
        log_msg(f"    ✓ Audio trích: {audio_size:.1f}MB", "DEBUG")
        
    except Exception as e:
        raise Exception(f"Extract audio failed: {str(e)}")

def load_model(lang_code, max_retries=3):
    global translator, sp_src, sp_tgt, current_lang
    
    if current_lang == lang_code and translator is not None:
        log_msg(f"    Model {lang_code} đã load", "DEBUG")
        return
    
    model_path = os.path.join(CT2_MODELS_FOLDER, f"{lang_code}-vi")
    source_spm = os.path.join(MODELS_FOLDER, f"{lang_code}-vi", "source.spm")
    target_spm = os.path.join(MODELS_FOLDER, f"{lang_code}-vi", "target.spm")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model: {model_path}")
    if not os.path.exists(source_spm):
        raise FileNotFoundError(f"Source SPM: {source_spm}")
    if not os.path.exists(target_spm):
        raise FileNotFoundError(f"Target SPM: {target_spm}")
    
    for attempt in range(max_retries):
        try:
            translator = ctranslate2.Translator(model_path, device="cuda")
            sp_src = sentencepiece.SentencePieceProcessor(model_file=source_spm)
            sp_tgt = sentencepiece.SentencePieceProcessor(model_file=target_spm)
            current_lang = lang_code
            log_msg(f"    ✓ Model {lang_code} tải thành công", "DEBUG")
            return
        except Exception as e:
            log_msg(f"    ⚠ Attempt {attempt+1}/{max_retries}: {str(e)}", "WARN")
            if attempt == max_retries - 1:
                raise
            time.sleep(2)

def process_file(file_path, whisper_model, file_index, total_files, batch_start_time):
    start_time = time.time()
    
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        if file_ext in SUBTITLE_EXTS:
            input_type = "srt"
        else:
            input_type = "video"
        
        log_msg(f"▶ [{file_index}/{total_files}] {os.path.basename(file_path)}", "INFO")
        
        # === BƯỚC 1: VIDEO → SRT ===
        if input_type == "video":
            step_start = time.time()
            log_msg(f"  [1/4] FFmpeg - Trích âm", "DEBUG")
            
            audio_path = os.path.join(os.path.dirname(file_path), f"{base_name}_temp.wav")
            apply_denoise = "111" in file_path
            
            extract_audio(file_path, audio_path, apply_denoise)
            step_time = time.time() - step_start
            log_msg(f"        ✓ {step_time:.1f}s", "DEBUG")
            
            # Whisper STT
            step_start = time.time()
            log_msg(f"  [2/4] Whisper - STT ({whisper_model})", "DEBUG")
            
            fw_model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
            segments, _ = fw_model.transcribe(audio_path, beam_size=5)
            segments_list = list(segments)
            
            srt_path = os.path.join(os.path.dirname(file_path), f"{base_name}.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for idx, segment in enumerate(segments_list, 1):
                    start = time.strftime('%H:%M:%S', time.gmtime(segment.start)) + f",{int((segment.start%1)*1000):03d}"
                    end = time.strftime('%H:%M:%S', time.gmtime(segment.end)) + f",{int((segment.end%1)*1000):03d}"
                    f.write(f"{idx}\n{start} --> {end}\n{segment.text.strip()}\n\n")
            
            step_time = time.time() - step_start
            log_msg(f"        ✓ {step_time:.1f}s ({len(segments_list)} subs)", "DEBUG")
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
            file_path = srt_path
        
        # === BƯỚC 2: ĐỌC SRT ===
        step_start = time.time()
        log_msg(f"  [3/4] CTranslate2 - Dịch", "DEBUG")
        
        subs = read_srt(file_path)
        sample_text = " ".join([s["text"] for s in subs[:5]])
        lang_code = detect_lang_regex(sample_text)
        log_msg(f"        Ngôn ngữ: {lang_code}", "DEBUG")
        
        load_model(lang_code)
        
        # === BƯỚC 3: DỊCH ===
        translated = []
        translate_start = time.time()
        
        for i, sub in enumerate(subs, 1):
            text = sub["text"]
            tokens = sp_src.encode(text, out_type=str)
            result = translator.translate_batch([tokens])
            translated_text = sp_tgt.decode(result[0].hypotheses[0])
            translated.append(translated_text)
            
            if i <= len(subs):
                elapsed = time.time() - translate_start
                avg_per_sub = elapsed / i
                remaining_translate = avg_per_sub * (len(subs) - i)
                
                if i % 20 == 0 or i == len(subs):
                    print(f"    [3/4] Dịch: [{i}/{len(subs)}] ⏱️ {format_time(remaining_translate)}", end="\r", flush=True)
        
        for i, sub in enumerate(subs):
            sub["text"] = translated[i]
        
        step_time = time.time() - step_start
        log_msg(f"        ✓ {step_time:.1f}s ({len(translated)} dịch)", "DEBUG")
        
        # === BƯỚC 4: CHIA DÀI + LƯU ===
        step_start = time.time()
        log_msg(f"  [4/4] Output - Chia & Lưu", "DEBUG")
        
        subs = split_long_subs(subs, MAX_SUB_LENGTH)
        output_file = os.path.join(os.path.dirname(file_path), f"{base_name}_vi.srt")
        write_srt(output_file, subs)
        
        step_time = time.time() - step_start
        log_msg(f"        ✓ {step_time:.1f}s ({len(subs)} subs)", "DEBUG")
        
        total_time = time.time() - start_time
        log_msg(f"  ✔ HOÀN TẤT ({total_time:.1f}s)", "INFO")
        
        # === HIỂN THỊ COUNTDOWN ===
        elapsed_total = time.time() - batch_start_time
        if file_index < total_files:
            avg_per_file = elapsed_total / file_index
            remaining_batch = avg_per_file * (total_files - file_index)
            print(f"\n📊 Tiến độ: [{file_index}/{total_files}] | ⏱️ Còn {format_time(remaining_batch)}\n")
        
        return True, total_time
    
    except Exception as e:
        total_time = time.time() - start_time
        log_msg(f"  ❌ LỖI ({total_time:.1f}s): {str(e)}", "ERROR")
        log_msg(f"TRACE:\n{traceback.format_exc()}", "ERROR")
        return False, total_time

# ======================== MAIN ========================
log_msg("=" * 110, "INFO")
log_msg("VIETSUB BATCH PROCESSOR v2.0", "INFO")
log_msg("=" * 110, "INFO")

print("\n1. Xử lý từ thư mục")
print("2. Xử lý file đơn lẻ")
choice = input("\n📋 Chọn (1-2): ").strip()

if choice == "1":
    input_path = input("📁 Thư mục input: ").strip().strip('"')
    if not os.path.isdir(input_path):
        log_msg(f"❌ Thư mục không tồn tại: {input_path}", "ERROR")
        sys.exit(1)
    
    files = [f for f in os.listdir(input_path) 
             if os.path.isfile(os.path.join(input_path, f)) 
             and os.path.splitext(f)[1].lower() in VIDEO_EXTS + SUBTITLE_EXTS]
    
elif choice == "2":
    input_path = input("📄 File input: ").strip().strip('"')
    if not os.path.isfile(input_path):
        log_msg(f"❌ File không tồn tại: {input_path}", "ERROR")
        sys.exit(1)
    files = [os.path.basename(input_path)]
    input_path = os.path.dirname(input_path)

else:
    log_msg("❌ Lựa chọn không hợp lệ", "ERROR")
    sys.exit(1)

whisper_opt = input("🎙️ Model Whisper (tiny/small/medium/large) [small]: ").strip() or "small"

log_msg(f"📁 Thư mục: {input_path}", "INFO")
log_msg(f"📦 Tổng files: {len(files)}", "INFO")
log_msg(f"🎙️ Whisper: {whisper_opt}", "INFO")
log_msg("", "INFO")

if len(files) == 0:
    log_msg("❌ Không tìm thấy file video/SRT", "ERROR")
    sys.exit(1)

batch_start = time.time()
stats = {"success": 0, "failed": 0}

for i, file in enumerate(files, 1):
    file_path = os.path.join(input_path, file)
    success, _ = process_file(file_path, whisper_opt, i, len(files), batch_start)
    
    if success:
        stats["success"] += 1
    else:
        stats["failed"] += 1

batch_total = time.time() - batch_start

log_msg("", "INFO")
log_msg("=" * 110, "INFO")
log_msg(f"TỔNG KẾT BATCH", "INFO")
log_msg("=" * 110, "INFO")
log_msg(f"✔ Thành công: {stats['success']}/{len(files)}", "INFO")
log_msg(f"❌ Thất bại: {stats['failed']}/{len(files)}", "INFO")
log_msg(f"⏱️ Tổng thời gian: {format_time(batch_total)}", "INFO")
log_msg(f"⏱️ Trung bình/file: {format_time(batch_total/len(files) if len(files) > 0 else 0)}", "INFO")
log_msg(f"📁 Log file: {log_file}", "INFO")
log_msg("=" * 110, "INFO")

print(f"\n✅ Hoàn tất! Log: {log_file}")
