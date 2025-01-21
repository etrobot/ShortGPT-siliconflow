import os
import subprocess
import time

import yt_dlp
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment

from shortGPT.audio.audio_duration import get_asset_duration

CONST_CHARS_PER_SEC = 20.5  # Arrived to this result after whispering a ton of shorts and calculating the average number of characters per second of speech.

WHISPER_MODEL = None



def downloadYoutubeAudio(url, outputFile):
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "no_call_home": True,
        "no_check_certificate": True,
        "format": "bestaudio/best", 
        "outtmpl": outputFile
    }

    attempts = 0
    max_attempts = 4
    while attempts < max_attempts:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                dictMeta = ydl.extract_info(
                    url,
                    download=True)
                if (not os.path.exists(outputFile)):
                    raise Exception("Audio Download Failed")
                return outputFile, dictMeta['duration']
        except Exception as e:
            attempts += 1
            if attempts == max_attempts:
                raise Exception(f"Failed downloading audio from the following video/url for url {url}", e.args[0])
            time.sleep(1)
            continue
    return None

def speedUpAudio(tempAudioPath, outputFile, expected_duration=None):
    tempAudioPath, duration = get_asset_duration(tempAudioPath, False)
    if not expected_duration:
        if (duration > 57):
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', tempAudioPath, '-af', f'atempo={(duration/57):.5f}', outputFile])
        else:
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', tempAudioPath, outputFile])
    else:
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', tempAudioPath, '-af', f'atempo={(duration/expected_duration):.5f}', outputFile])
    if (os.path.exists(outputFile)):
        return outputFile

def ChunkForAudio(alltext, chunk_size=2500):
    alltext_list = alltext.split('.')
    chunks = []
    curr_chunk = ''
    for text in alltext_list:
        if len(curr_chunk) + len(text) <= chunk_size:
            curr_chunk += text + '.'
        else:
            chunks.append(curr_chunk)
            curr_chunk = text + '.'
    if curr_chunk:
        chunks.append(curr_chunk)
    return chunks


def _create_speech_recognizer(audio_path, start_time=0):
    """创建并配置语音识别器"""
    subscription_key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    
    if not subscription_key or not region:
        raise ValueError("Azure Speech credentials not found")
    
    speech_config = speechsdk.SpeechConfig(
        subscription=subscription_key,
        region=region
    )
    
    # 修正属性设置，使用正确的 PropertyId 枚举
    speech_config.set_property(
        speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "3000"
    )
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "5000"
    )
    speech_config.set_property(
        speechsdk.PropertyId.Conversation_Initial_Silence_Timeout, "5000"
    )
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "5000"
    )
    
    audio_input = speechsdk.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_input,
        language="zh-CN"
    )
    
    results = []
    done = False
    
    def handle_result(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            start = (evt.result.offset / 10000000) + start_time
            duration = evt.result.duration / 10000000
            end = start + duration
            
            segment = ((start, end), evt.result.text)
            results.append(segment)
            print(f"[DEBUG] 识别片段: {start:.2f}s - {end:.2f}s: {evt.result.text}")
    
    def stop_cb(evt):
        nonlocal done
        done = True
    
    recognizer.recognized.connect(handle_result)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)
    
    return recognizer, results, lambda: done

def audioToText(filename, model_size="base"):
    """主要的音频转文本函数"""
    try:
        print(f"[DEBUG] 开始处理音频文件: {filename}")
        
        # 预处理音频
        audio = AudioSegment.from_file(filename)
        total_duration = len(audio) / 1000.0
        print(f"[INFO] 音频总长度: {total_duration:.2f}秒")
        
        # 确保音频质量
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        temp_audio_path = filename + "_temp.wav"
        audio.export(temp_audio_path, format="wav")
        
        # 开始识别
        recognizer, all_results, is_done = _create_speech_recognizer(temp_audio_path)
        recognizer.start_continuous_recognition()
        
        while not is_done():
            time.sleep(.5)
        recognizer.stop_continuous_recognition()
        
        # 检查识别完整性
        if len(all_results) > 0:
            last_timestamp = all_results[-1][0][1]
            coverage_percentage = (last_timestamp / total_duration) * 100
            print(f"[INFO] 识别覆盖率: {coverage_percentage:.2f}%")
            
            if coverage_percentage < 95:
                print(f"[WARNING] 识别不完整，处理剩余部分...")
                remaining_results = process_remaining_audio(audio, last_timestamp, total_duration, temp_audio_path)
                if remaining_results:
                    all_results.extend(remaining_results)
                    all_results.sort(key=lambda x: x[0][0])
        
        # 清理临时文件
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        # 构建返回结果
        result = {
            "segments": [
                {
                    "start": start,
                    "end": end,
                    "text": text
                }
                for (start, end), text in all_results
            ],
            "language": "zh-CN",
            "text": " ".join(text for _, text in all_results)
        }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] audioToText 发生错误: {str(e)}")
        return None

def process_remaining_audio(audio, last_timestamp, total_duration, original_path):
    """处理未识别的音频部分"""
    try:
        print(f"[INFO] 处理剩余音频: {last_timestamp:.2f}s - {total_duration:.2f}s")
        
        # 截取剩余音频
        start_ms = int(last_timestamp * 1000) - 1000
        remaining_audio = audio[start_ms:]
        
        # 导出剩余部分
        remaining_path = original_path.replace('.wav', '_remaining.wav')
        remaining_audio.export(remaining_path, format="wav")
        
        # 使用共同的识别器创建函数
        recognizer, results, is_done = _create_speech_recognizer(remaining_path, last_timestamp - 1)
        
        # 开始识别
        recognizer.start_continuous_recognition()
        while not is_done():
            time.sleep(.5)
        recognizer.stop_continuous_recognition()
        
        # 清理临时文件
        if os.path.exists(remaining_path):
            os.remove(remaining_path)
        
        print(f"[INFO] 剩余部分识别完成，识别了 {len(results)} 个片段")
        return results
        
    except Exception as e:
        print(f"[ERROR] 处理剩余音频时出错: {str(e)}")
        return []

def getSpeechBlocks(whispered_result, silence_time=0.8):
    """
    从语音识别结果中提取语音块
    """
    print(f"[DEBUG] 开始处理语音块，silence_time={silence_time}")
    
    if not whispered_result or not isinstance(whispered_result, dict):
        print("[ERROR] 无效的识别结果")
        return []
        
    try:
        segments = whispered_result.get('segments', [])
        print(f"[DEBUG] 输入包含 {len(segments)} 个语音片段")
        
        speech_blocks = []
        current_block = []
        
        for i, segment in enumerate(segments):
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            if not text:
                continue
            
            # For first segment or if time gap is small enough, add to current block
            if not current_block:
                current_block.append((start, end, text))
            else:
                prev_end = current_block[-1][1]
                time_gap = start - prev_end
                
                if time_gap < silence_time:
                    current_block.append((start, end, text))
                else:
                    # Create new block when silence is detected
                    block_start = current_block[0][0]
                    block_end = current_block[-1][1]
                    block_text = ' '.join(seg[2] for seg in current_block)
                    speech_blocks.append([[block_start, block_end], block_text])
                    current_block = [(start, end, text)]
        
        # Handle the last block
        if current_block:
            block_start = current_block[0][0]
            block_end = current_block[-1][1]
            block_text = ' '.join(seg[2] for seg in current_block)
            speech_blocks.append([[block_start, block_end], block_text])
        
        print(f"[DEBUG] 处理完成，生成了 {len(speech_blocks)} 个语音块")
        for i, block in enumerate(speech_blocks):
            print(f"[DEBUG] 语音块 {i+1}: {block[0][0]:.2f}s - {block[0][1]:.2f}s: {block[1]}")
        return speech_blocks
        
    except Exception as e:
        print(f"[ERROR] getSpeechBlocks 发生错误: {str(e)}")
        return []

def getWordsPerSec(filename):
    a = audioToText(filename)
    return len(a['text'].split()) / a['segments'][-1]['end']


def getCharactersPerSec(filename):
    a = audioToText(filename)
    return len(a['text']) / a['segments'][-1]['end']

def run_background_audio_split(sound_file_path):
    try:
        # Run spleeter command
        # Get absolute path of sound file 
        output_dir = os.path.dirname(sound_file_path)
        command = f"spleeter separate -p spleeter:2stems -o '{output_dir}' '{sound_file_path}'"

        process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # If spleeter runs successfully, return the path to the background music file
        if process.returncode == 0:
            return os.path.join(output_dir, sound_file_path.split("/")[-1].split(".")[0], "accompaniment.wav")
        else:
            return None
    except Exception:
        # If spleeter crashes, return None
        return None
